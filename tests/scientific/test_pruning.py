import copy
import math
import pytest
import torch
from bagel_robotics._internal.adaptive_mask.geometry import Geometry, ParameterInventory
from bagel_robotics._internal.adaptive_mask.export import build_compact,sliced_state
from bagel_robotics._internal.adaptive_mask.masking import install
from bagel_robotics.models.backbone.wan21_14b.wan_video_dit import WanVideoDiT
from bagel_robotics.adaptive_pruning.masks import Masks,SoftCost,ranked_indices,binary_indices,schedule
from bagel_robotics._internal.training.coefficient import Coefficient
from bagel_robotics.adaptive_pruning.losses import reliability,objectives
from bagel_robotics.adaptive_pruning.selection import readiness,jaccard,candidate_key
from bagel_robotics.adaptive_pruning.data import positions,coverage
from bagel_robotics.data_selection.manifest import BalancedStream


def geometry():return Geometry(hidden=80,ffn=128,layers=3,heads=4,head_dim=8,pack=2,hidden_floor=76,ffn_floor=16,head_floor=1)
def kwargs():return dict(hidden_dim=80,ffn_dim=128,num_layers=3,num_heads=4,attn_head_dim=8,freq_dim=8,text_dim=10,in_dim=8,out_dim=2,cross_attn_norm=True)
def masks():
    g=geometry()
    return Masks(g,dict(hidden_scores=torch.arange(80.),ffn_scores=torch.arange(256.).reshape(2,128),cross_scores=torch.arange(8.).reshape(2,4)),
                 dict(hidden_scores=76,ffn_scores=64,cross_scores=3))


def test_rank_initialization_and_ties():
    c=masks();assert (c.hidden_scores>0).sum()==76
    assert torch.all((c.hidden_scores==1.5)|(c.hidden_scores==-1.5))
    assert ranked_indices(torch.zeros(5),2)==[0,1]
    assert binary_indices(torch.tensor([.1,.2,.3,-1.,-2.,-3.]),2,2)==([0,1,2,3],1)


def test_soft_is_continuous_not_ste_and_rounded_export_is_shared():
    c=masks()
    assert all(torch.all(g==1) for g in c.gates(0))
    gates=c.gates(200);assert 0<float(gates[0].detach().min())<float(gates[0].detach().max())<1
    sum(x.sum() for x in gates).backward();assert all(p.grad.abs().min()>0 for p in c.parameters())
    with torch.no_grad():c.hidden_scores.fill_(-2);c.ffn_scores.fill_(-2);c.cross_scores.fill_(-2)
    s,restored=c.selection();assert len(s['hidden_indices'])==76
    assert s['ffn_indices'][-1]==list(range(128))
    assert restored['hidden']==76 and len(s['cross_head_indices'][0])==1


@pytest.mark.parametrize('u,alpha,temp,ent',[(0,0,1,0),(100,.5,1,0),(200,1,1,0),(500,1,1,0),(750,1,5.5,.005),(1000,1,10,.01),(3000,1,10,.01)])
def test_schedules(u,alpha,temp,ent):
    s=schedule(u);assert (s['alpha'],s['temperature'],s['entropy'])==(alpha,temp,ent)


def test_cost_matches_exact_slicer_at_multiple_binary_masks():
    with torch.device('meta'):model=WanVideoDiT(**kwargs())
    inv=ParameterInventory.from_model(model,geometry());cost=SoftCost(inv);c=masks()
    for seed in range(7):
        torch.manual_seed(seed)
        with torch.no_grad():
            for p in c.parameters():p.copy_(torch.randn_like(p)+seed-3)
        s,_=c.selection();assert float(cost.parameters(c.gates(200,'binary')))==inv.count(s)
    value=cost.loss(c.gates(200));grads=torch.autograd.grad(value,c.families())
    assert all(torch.isfinite(g).all() and (g>0).all() for g in grads)


def test_cost_changes_correspond_to_deleted_tensors():
    with torch.device('meta'):model=WanVideoDiT(**kwargs())
    inv=ParameterInventory.from_model(model,geometry());s=geometry().full();s['hidden_indices']=list(range(76))
    before=inv.count(s);s['ffn_indices'][0]=list(range(64))
    assert before-inv.count(s)==64*(2*76+1)
    before=inv.count(s);s['cross_head_indices'][0]=[0,1,2]
    assert before-inv.count(s)==8*(6*76+8)


def test_coefficient_is_capped_ramped_and_resumable():
    c=Coefficient(cap=1.)
    for u in range(193,201):c.observe(u,[100,200,300],[1,1,1])
    assert c.measured==10 and c.applied==1 and c.weight(200)==0
    assert c.weight(250)==.5 and c.weight(300)==1 and c.weight(3000)==1
    restored=Coefficient.from_state(c.state_dict(),1)
    assert restored.state_dict()==c.state_dict()
    with pytest.raises(ValueError):Coefficient.from_state(c.state_dict(),.5)


def test_small_measured_value_not_raised_to_cap():
    c=Coefficient()
    for u in range(193,201):c.observe(u,[.01,.02,.03],[1,1,1])
    assert c.applied==pytest.approx(.001)


@pytest.mark.parametrize('bad',[0.,float('nan'),float('inf'),1e-13])
def test_invalid_coefficient_never_falls_back(bad):
    c=Coefficient()
    for u in range(193,233):c.observe(u,[1,bad,1],[1,1,1])
    assert c.unresolved and c.applied is None and c.weight(300)==0


def test_replacement_measurements_and_repeated_update():
    c=Coefficient()
    for u in range(193,201):c.observe(u,[0 if u==193 else 1,1,1],[1,1,1])
    assert c.applied is None
    c.observe(201,[2,1,1],[1,1,1]);assert c.resolved_update==201
    with pytest.raises(ValueError):c.observe(201,[1]*3,[1]*3)


@pytest.mark.parametrize('cap',[0.,-1.,float('nan'),float('inf')])
def test_invalid_cap(cap):
    with pytest.raises(ValueError):Coefficient(cap=cap)


def test_relative_reliability_and_shared_component_veto():
    t=torch.tensor([1.,2.,0.,0.,1.],requires_grad=True)
    s=torch.tensor([2.,1.,0.,2.,2.],requires_grad=True);zero=torch.ones(5)*10
    tc=torch.zeros(5,3);zc=torch.ones(5,3)*10;tc[-1,2]=20
    r,_=reliability(t,s,zero,tc,zc)
    torch.testing.assert_close(r,torch.tensor([.5,0.,0.,1.,0.]));assert not r.requires_grad


def test_zero_baseline_veto():
    r,_=reliability(torch.tensor([2.]),torch.tensor([4.]),torch.tensor([1.]),torch.zeros(1,3),torch.ones(1,3))
    assert r.item()==0


def test_losses_keep_original_batch_denominator_and_rejected_gt():
    student=torch.tensor([[[2.]],[[1.]]],requires_grad=True)
    teacher=torch.tensor([[[1.]],[[2.]]]);target=torch.ones_like(student)*3
    video=torch.ones(2,1,5,2,2,requires_grad=True);tv=torch.zeros_like(video)
    kv=torch.ones(2,40,2,requires_grad=True)
    loss,metrics,details,terms=objectives(student,teacher,target,None,torch.ones(2),video,tv,kv,torch.ones(2,2),1)
    # First teacher is worse -> rejected; second teacher improves 4 -> 1 -> r=.75.
    torch.testing.assert_close(details['reliability'],torch.tensor([0.,.75]))
    assert terms['kv'].item()==pytest.approx(.375) and terms['video'].item()==pytest.approx(.375)
    loss.backward();assert student.grad[0].abs().sum()>0
    assert kv.grad[0].abs().sum()==0 and kv.grad[:, :20].abs().sum()==0
    assert video.grad[0].abs().sum()==0 and video.grad[:,:,:3].abs().sum()==0


def test_history_padding_video_not_supervised():
    student=torch.ones(1,1,5,2,2,requires_grad=True)
    from bagel_robotics._internal.adaptive_mask.losses import video_error
    value,valid=video_error(student,torch.zeros_like(student),torch.tensor([[1.,0.]]))
    value.sum().backward();assert student.grad[:,:,4].abs().sum()==0 and valid.all()


def test_partition_and_resume_sequence_independent_of_gpu_batch():
    flat=lambda micro:sorted(p for rank in range(8) for b in positions(2,rank,8,micro) for p in b)
    assert flat(1)==flat(2)==flat(4)==list(range(256,512))
    rows=[dict(id=i,task='a' if i<3 else 'b') for i in range(8)]
    a=BalancedStream(rows,42);b=BalancedStream(rows,42)
    assert [a.at(i) for i in range(17,45)]==[b.at(i) for i in range(17,45)]
    for task,size in (('a',3),('b',5)):
        chosen=[a.at(i)['id'] for i in range(size*2) if a.at(i)['task']==task]
        assert len(chosen)==len(set(chosen))==size
    assert min(x['minimum_visits'] for x in coverage(a,20).values())==2


def history():
    s=masks().selection()[0]
    return [dict(update=u,selection=copy.deepcopy(s),parameters=100000,
        regularizer_state=[10,.01,.01,.2],gradient_active=True,
        actions={'task/precision':dict(student_error=2.,teacher_error=1.),'suite/precision':dict(student_error=2.,teacher_error=1.)})
        for u in (1000,1250,1500,1750,2000)]


def test_no_teacher_closeness_gate_or_soft_binary_gate():
    h=history();assert readiness(h,geometry(),2000,2)['status']=='ready_for_recovery'
    assert jaccard([],[])==1


@pytest.mark.parametrize('problem',['coverage','identity','capacity','gradient','temperature','deterioration'])
def test_readiness_rejects_unstable_not_large_teacher_gap(problem):
    h=history();cov=2
    if problem=='coverage':cov=1
    elif problem=='identity':h[-1]['selection']['ffn_indices'][0]=list(range(64))
    elif problem=='capacity':h[0]['parameters']=120000
    elif problem=='gradient':h[-1]['gradient_active']=False
    elif problem=='temperature':h[0]['regularizer_state'][0]=9
    else:
        for r in h[-2:]:r['actions']['task/precision']['student_error']=3
    assert readiness(h,geometry(),2000,cov)['status']=='selection_not_converged'


def test_trained_binary_matches_physically_sliced_network_and_masks_receive_action_gradient():
    torch.manual_seed(42)
    wide=WanVideoDiT(**kwargs()).float()
    # Explicitly change student tensors: surgery must preserve trained weights.
    with torch.no_grad():wide.blocks[0].ffn[0].weight.add_(.02)
    c=masks();s,_=c.selection();small=build_compact(kwargs(),s,geometry(),device='cpu').float()
    state,_=sliced_state(wide.state_dict(),small,s,geometry());small.load_state_dict(state)
    branch=install(wide);compact=install(small)
    inputs=dict(context=torch.randn(1,3,10),context_mask=torch.ones(1,3,dtype=torch.bool),
        clip_fea=torch.randn(1,257,1280),condition_latents=torch.randn(1,2,1,4,4))
    noise=torch.randn(1,2,2,4,4);time=torch.tensor([1000.]);attention=torch.ones(8,8,dtype=torch.bool)
    a=branch(inputs,noise,time,attention,*c.gates(200,'binary'));b=compact(inputs,noise,time,attention)
    for left,right in [(a['video'],b['video'])]+[(x[k],y[k]) for x,y in zip(a['cache'],b['cache']) for k in ('k','v')]:
        rel=(left-right).square().mean().sqrt()/right.square().mean().sqrt().clamp_min(1e-6)
        assert rel<2e-5
    # Small frozen downstream head: gradients must cross it into both masks/video.
    head=torch.nn.Linear(32,2).requires_grad_(False)
    out=branch(inputs,noise,time,attention,*c.gates(200))
    predicted=head(out['cache'][-1]['v']).square().mean();predicted.backward()
    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in c.parameters())
    assert wide.blocks[0].ffn[0].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in head.parameters())


def test_native_loader_builds_same_compact_shapes_without_mask_modules():
    from bagel_robotics.models.backbone.wan21_14b.helpers.compact_channels import build
    g=geometry();s=masks().selection()[0]
    reference=build_compact(kwargs(),s,g,device='meta')
    spec=dict(schema='vpp2-compact-channels-v1',geometry=g.__dict__,selection=s,
              model_kwargs=kwargs(),parameters=sum(p.numel() for p in reference.parameters()))
    native=build(spec,device='meta')
    assert type(native) is WanVideoDiT
    assert {k:tuple(v.shape) for k,v in native.state_dict().items()}=={k:tuple(v.shape) for k,v in reference.state_dict().items()}
    assert all('Mask' not in type(m).__name__ for m in native.modules())
    spec['parameters']-=1
    with pytest.raises(ValueError):build(spec,device='meta')


def test_coefficient_uses_global_gradient_not_average_local_norm():
    # Opposing rank gradients must cancel before the strength is estimated.
    rank_gradients=torch.tensor([[1.,2.],[-1.,-2.]])
    norm=float(rank_gradients.mean(0).norm())
    assert norm==0 and rank_gradients.norm(dim=-1).mean()>0
    c=Coefficient()
    c.observe(193,[norm,1,1],[1,1,1])
    assert not c.samples['hidden']


def test_episode_windows_never_reuse_old_frame_identity():
    from bagel_robotics.adaptive_pruning.data import EpisodeWindows
    rows=[dict(episode=7,length=40,frame=0,id=0,task='a'),dict(episode=9,length=35,frame=0,id=1,task='a')]
    windows=EpisodeWindows(rows)
    assert len(windows)==13
    assert windows[8]['episode']==7 and windows[8]['frame']==8
    assert windows[9]['episode']==9 and windows[9]['frame']==0
    assert len({windows[i]['id'] for i in range(13)})==13
    assert windows[8]['history_indices'][-1]==8


def test_aggregate_uses_task_balanced_suite_means():
    from bagel_robotics.adaptive_pruning.selection import aggregate
    rows=[dict(window_id=i,task='a',suite='precision',student_error=1.,teacher_error=1.) for i in range(8)]
    rows.append(dict(window_id=8,task='b',suite='precision',student_error=3.,teacher_error=1.))
    assert aggregate(rows)['suite/precision']['student_error']==2


def test_stopping_checks_first_to_last_removed_channel_drift():
    from bagel_robotics.adaptive_pruning.selection import overlaps
    g=Geometry(hidden=5120,ffn=13824,layers=2,heads=40,head_dim=128,pack=128,hidden_floor=4864,ffn_floor=4096,head_floor=8)
    a=g.full();a['hidden_indices']=list(range(4864))
    b=copy.deepcopy(a);b['hidden_indices']=list(range(20,4884))
    report=overlaps(a,b,g)
    h=next(x for x in report if x['family']=='hidden_indices')
    assert not h['passed'] and h['removed']<.95


def test_cap_configuration_has_no_silent_overrides():
    from pathlib import Path
    from bagel_robotics.adaptive_pruning.config import load,validate
    c=load(Path(__file__).resolve().parents[2]/'configs/recipes/mask_learning.json')
    c['size_coefficient']['upper_cap']=.5;validate(c)
    c['size_coefficient']['upper_cap']=1.01
    with pytest.raises(ValueError):validate(c)
    c['size_coefficient']['upper_cap']=1
    c['train']['frozen_action']=False
    with pytest.raises(ValueError):validate(c)


def test_gradient_probe_restores_trainability_and_does_not_accumulate():
    from bagel_robotics.adaptive_pruning.gradient_probe import frozen_weight_mask_gradient
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.randn(3,3))
        def forward(self,x,mask):return (x@self.weight)*mask
    m=Tiny();gate=torch.nn.Parameter(torch.ones(3));x=torch.randn(2,3)
    # Nonempty accumulated gradients must survive the diagnostic exactly.
    m.weight.grad=torch.ones_like(m.weight);gate.grad=torch.full_like(gate,7)
    rng=torch.get_rng_state().clone()
    measured=frozen_weight_mask_gradient(m,[gate],lambda y:y.square().mean(),(x,gate))[0]
    expected=torch.autograd.grad(m(x,gate).square().mean(),gate)[0]
    torch.testing.assert_close(measured,expected,rtol=0,atol=0)
    assert m.weight.requires_grad and torch.all(m.weight.grad==1) and torch.all(gate.grad==7)
    assert torch.equal(rng,torch.get_rng_state())
    with pytest.raises(RuntimeError):
        frozen_weight_mask_gradient(m,[gate],lambda y:(_ for _ in ()).throw(RuntimeError('test')),(x,gate))
    assert m.weight.requires_grad


def test_rng_diagnostics_restore_python_numpy_and_torch():
    import random,numpy as np
    from bagel_robotics._internal.training.rng import diagnostic_rng,rng_state
    from bagel_robotics.adaptive_pruning.preflight import fingerprint
    before=fingerprint(rng_state())
    with diagnostic_rng():
        random.random();np.random.rand();torch.rand(3)
    assert before==fingerprint(rng_state())


def test_logging_preserves_training_randomness():
    import random,numpy as np
    from bagel_robotics.adaptive_pruning.report import isolated_logging
    from bagel_robotics._internal.training.rng import rng_state
    from bagel_robotics.adaptive_pruning.preflight import fingerprint
    before=fingerprint(rng_state())
    @isolated_logging
    def logger():
        random.random();np.random.rand();torch.rand(3)
    logger();assert before==fingerprint(rng_state())


def test_checkpointed_mask_diagnostic_preserves_trainability_and_skips_weight_hooks():
    from torch.utils.checkpoint import checkpoint
    from bagel_robotics.adaptive_pruning.gradient_probe import frozen_weight_mask_gradient
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.layer=torch.nn.Linear(4,4)
        def forward(self,x,mask):
            assert self.layer.weight.requires_grad
            return checkpoint(lambda v:self.layer(v*mask).sin(),x,use_reentrant=False)
    model=Model(); mask=torch.nn.Parameter(torch.ones(4)); x=torch.randn(3,4)
    invoked=[]
    handles=[p.register_post_accumulate_grad_hook(lambda p:invoked.append(p)) for p in model.parameters()]
    expected=torch.autograd.grad(model(x,mask).square().mean(),mask)[0]
    values=frozen_weight_mask_gradient(model,[mask],lambda y:dict(gt=y.square().mean(),video=y.mean(),kv=y.abs().mean()),(x,mask))
    torch.testing.assert_close(values['gt'][0],expected,rtol=0,atol=0)
    assert invoked==[] and all(p.grad is None and p.requires_grad for p in model.parameters())
    assert mask.grad is None
    for handle in handles:handle.remove()




def test_mask_diagnostic_keeps_saved_sharded_weight_values_after_release():
    from bagel_robotics.adaptive_pruning.gradient_probe import frozen_weight_mask_gradient
    class ReleasedProjection(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight=torch.nn.Parameter(torch.randn(4,4))
            self.weight.ds_id=7; self.weight.ds_numel=self.weight.numel()
        def forward(self,x,mask):
            y=(x*mask)@self.weight
            # Mimic ZeRO releasing an uncheckpointed conditioning projection.
            self.weight.data=torch.empty(0)
            return y
    module=ReleasedProjection(); weight=module.weight.detach().clone()
    x=torch.randn(3,4);mask=torch.nn.Parameter(torch.ones(4))
    expected=torch.autograd.grad(((x*mask)@weight).square().mean(),mask)[0]
    actual=frozen_weight_mask_gradient(module,[mask],lambda y:y.square().mean(),(x,mask))[0]
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert module.weight.grad is None and mask.grad is None


def test_multi_objective_diagnostic_releases_untraversed_saved_activation_graphs():
    import gc,weakref
    from bagel_robotics.adaptive_pruning.gradient_probe import frozen_weight_mask_gradient
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.layer=torch.nn.Linear(4,4);self.saved=[]
        def forward(self,x,mask):
            base=self.layer(x*mask)
            video=base.tanh()
            self.saved.append(weakref.ref(video))
            return dict(video=video,cache=base*2)
    model=Model();mask=torch.nn.Parameter(torch.ones(4));x=torch.randn(2,4)
    enabled=gc.isenabled();gc.disable()
    try:
        for _ in range(4):
            frozen_weight_mask_gradient(model,[mask],lambda y:dict(
                gt=y['cache'].square().mean(),video=y['video'].square().mean(),kv=y['cache'].abs().mean()),(x,mask))
        assert all(ref() is None for ref in model.saved), 'Saved-tensor hooks retained activation/grad_fn reference cycles'
    finally:
        if enabled:gc.enable()
        gc.collect()
