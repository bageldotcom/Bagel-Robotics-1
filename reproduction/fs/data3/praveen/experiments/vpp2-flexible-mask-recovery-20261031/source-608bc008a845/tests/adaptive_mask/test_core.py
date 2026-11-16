import copy
import pytest
import torch
from vpp2.adaptive_mask.geometry import Geometry,ParameterInventory
from vpp2.adaptive_mask.controller import Controller,straight_through
from vpp2.adaptive_mask.masking import install
from vpp2.adaptive_mask.export import build_compact,sliced_state
from vpp2.adaptive_mask.reliability import weights
from vpp2.adaptive_mask.losses import objectives
from vpp2.adaptive_mask.quality import assess,stable
from vpp2.adaptive_mask.budget import Budget
from vpp2.models.wan21_14b.wan_video_dit import WanVideoDiT
from vpp2.compact_kv.blocks import install_blocks

torch.set_num_threads(1)

def small():
    g=Geometry(hidden=12,ffn=20,layers=3,heads=2,head_dim=8,pack=2,hidden_floor=6,ffn_floor=4,head_floor=1)
    args=dict(hidden_dim=12,ffn_dim=20,num_layers=3,num_heads=2,attn_head_dim=8,freq_dim=8,text_dim=10,in_dim=8,out_dim=2,cross_attn_norm=True)
    torch.manual_seed(100008);m=WanVideoDiT(**args).float().eval()
    inp=dict(context=torch.randn(1,3,10),context_mask=torch.ones(1,3,dtype=torch.bool),clip_fea=torch.randn(1,257,1280),condition_latents=torch.randn(1,2,1,4,4))
    return g,args,m,(inp,torch.randn(1,2,2,4,4),torch.tensor([1000.]),torch.ones(8,8,dtype=torch.bool))

def native_outputs(m,data):
    inp,noise,t,mask=data
    pre=m.pre_dit(noise,t,inp['context'],inp['context_mask'],inp['clip_fea'],inp['condition_latents'])
    x=pre['tokens'];cache=[]
    for b in m.blocks:
        x,k,v=b(x,pre['t_mod'],pre['freqs'],pre['context'],pre['context_mask'],mask);cache.append(dict(k=k,v=v))
    return dict(cache=cache,video=m.post_dit(x,pre))

def test_open_full_video_and_all_kv_match_native():
    g,args,m,data=small();branch=install(copy.deepcopy(m)).eval();install_blocks(m,())
    with torch.no_grad():a=native_outputs(m,data);b=branch(*data)
    torch.testing.assert_close(a['video'],b['video'],rtol=0,atol=0)
    for x,y in zip(a['cache'],b['cache']):
        for k in x:torch.testing.assert_close(x[k],y[k],rtol=0,atol=0)

def test_binary_sliced_model_full_path_and_gradients():
    g,args,m,data=small();branch=install(copy.deepcopy(m)).eval().requires_grad_(False)
    select=g.full();select['hidden_indices']=[0,2,4,6,8,10]
    select['ffn_indices']=[list(range(0,20,2)),list(range(6)),list(range(20))]
    select['cross_head_indices']=[[1],[0],[0,1]]
    c=Controller(g);c.set_selection(select)
    compact=build_compact(args,select,g,device='cpu').float().eval()
    state,audit=sliced_state(m.state_dict(),compact,select,g);compact.load_state_dict(state,strict=True);install_blocks(compact,())
    inv=ParameterInventory.from_model(m,g)
    expected=sum(p.numel() for p in compact.parameters())
    assert inv.count(select)==expected==inv.cost(6,[10,6,20],[1,1,2])
    with torch.no_grad():
        a=branch(*data,*c.gates(mode='binary'));b=native_outputs(compact,data)
    torch.testing.assert_close(a['video'],b['video'],rtol=2e-5,atol=3e-6)
    for x,y in zip(a['cache'],b['cache']):
        for k in x:torch.testing.assert_close(x[k],y[k],rtol=2e-5,atol=3e-6)
    branch.train();branch.video.use_gradient_checkpointing=True
    out=branch(*data,*c.gates())
    (out['video'].square().mean()+sum(v.square().mean() for layer in out['cache'] for v in layer.values())).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in c.parameters())
    assert c.ffn_scores.grad[0,1::2].abs().sum()>0
    assert c.cross_scores.grad[0,0].abs()>0 # removed head can return
    assert c.hidden_scores.grad[1::2].abs().sum()>0
    assert not any(p.grad is not None for p in branch.parameters())

def test_projection_respects_physical_budget_and_deterministic_export():
    g,args,m,_=small();inv=ParameterInventory.from_model(m,g);c=Controller(g)
    cap=inv.cost(10,[12,10,20],[1,1,2]);a=c.project(inv,cap,10);s=c.selection()
    assert a['parameters']==inv.count(s)<=cap
    c.project(inv,cap,10);assert s==c.selection()
    assert all(torch.equal(t.detach(),b) for t,b in zip(c.gates(),c.gates(mode='binary')))
    d=Controller(g);d.load_state_dict(c.state_dict());assert d.selection()==s
    # Capacity exchanges allowed, layer masks not shared.
    with torch.no_grad():c.ffn_scores[0].fill_(20);c.ffn_scores[1].fill_(-20)
    c.project(inv,cap,10);assert len(c.selection()['ffn_indices'][0])>=len(c.selection()['ffn_indices'][1])

def test_removed_unit_can_regrow():
    c=Controller(Geometry(hidden=2,ffn=4,layers=2,heads=2,head_dim=2,pack=1,hidden_floor=1,ffn_floor=1,head_floor=1))
    with torch.no_grad():c.hidden_scores.copy_(torch.tensor([-.01,.01]))
    hard=torch.tensor([0.,1.]);g=straight_through(c.hidden_scores,hard)
    loss=(g[0]-1).square();loss.backward()
    with torch.no_grad():c.hidden_scores-=c.hidden_scores.grad
    assert c.hidden_scores.argmax()==0

def test_reliability_shared_continuous_veto_detached():
    t=torch.tensor([.1,.5,1.,.1,.1],requires_grad=True);s=torch.tensor([.1,1.,1.,.01,1.]);z=torch.tensor([1.,1.,1.,1.,.01])
    w,r=weights(t,s,z)
    torch.testing.assert_close(w,torch.tensor([.9,.5,0.,0.,0.]));assert not w.requires_grad
    assert r['base'][3]>.8 and r['student_veto'][3]
    with pytest.raises(FloatingPointError):weights(t*float('nan'),s,z)

def test_losses_zero_reliability_keeps_gt_and_no_early_kv_grad():
    p=torch.ones(2,3,2,requires_grad=True);teacher=torch.full_like(p,10);target=torch.zeros_like(p)
    kv=torch.ones(2,40,2,requires_grad=True);sv=torch.ones(2,1,5,2,2,requires_grad=True);tv=torch.zeros_like(sv)
    loss,metrics,details,terms=objectives(p,teacher,target,None,torch.ones(2),sv,tv,kv,torch.ones(2,2),100)
    loss.backward();assert p.grad.abs().sum()>0 and kv.grad.abs().sum()==0 and sv.grad.abs().sum()==0
    assert metrics['loss/total']==metrics['loss/gt_action']
    p=torch.full((2,3,2),2.,requires_grad=True);teacher=torch.ones_like(p);kv=torch.ones(2,40,2,requires_grad=True)
    loss,_,d,_=objectives(p,teacher,target,None,torch.ones(2),sv,tv,kv,torch.ones(2,2),100)
    # Zero-flow veto rejects when target is zero; use nonzero flow target.
    target=torch.full_like(p,4.);teacher=torch.full_like(p,3.5)
    loss,_,d,_=objectives(p,teacher,target,None,torch.ones(2),sv,tv,kv,torch.ones(2,2),100)
    loss.backward();assert kv.grad[:,:20].count_nonzero()==0 and kv.grad[:,20:].count_nonzero()==80
    torch.testing.assert_close(kv.grad[:,20:],torch.full_like(kv.grad[:,20:],.05*(1-.25/16)/80))

def test_quality_requires_positive_evidence_not_absence_of_significance():
    rows=[dict(task='x',suite='s',episode=i,frame=0,teacher_error=1.,student_error=v) for i,v in enumerate([.2,1.8,.2,1.8])]
    assert not assess(rows)['passed']
    for r in rows:r['student_error']=1.001
    assert assess(rows)['passed']
    for r in rows:r['student_error']=1.03
    assert not assess(rows)['passed'] # suite 2% tighter than task 5%

def test_single_progressive_schedule_requires_fitting_authorization():
    b=Budget(original=16_394_878_784)
    checks=[];caps=[]
    while b.attempt<1960:
        b.advance();caps.append(b.cap)
        if b.full_due():
            checks.append(b.attempt)
            result=b.decide(True,stable=b.attempt==1960,coverage=True)
        if b.attempt%70==0:
            b.observe_fitting(dict(source='fitting_only',complete=True,passed=True,resolved_violation=False),
                              dict(ffn_max=0.,cross_heads_max=0.))
    assert checks==[280,560,1120,1680,1960]
    assert result=='finish' and b.cap==10_000_000_000 and not b.extra_used
    assert all(a>=c for a,c in zip(caps,caps[1:]))
    assert caps[69]==b.original and caps[559]==14_000_000_000


def test_one_stability_extension_and_resume_schedule_state():
    b=Budget(original=16_394_878_784)
    for _ in range(1960):b.advance()
    assert b.decide(True,stable=False,coverage=True)=='extend_final_check_once'
    resumed=Budget(**b.state_dict())
    for _ in range(280):b.advance();resumed.advance()
    assert b.state_dict()==resumed.state_dict() and b.attempt==b.maximum_updates
    assert b.decide(True,stable=False,coverage=True)=='stop_not_stable'


def test_hidden_protection_filters_optimizer_and_sync():
    from vpp2.adaptive_mask.controller import synchronize_controller
    g,args,m,_=small();c=Controller(g,learn_hidden=False)
    sum(t.sum() for t in c.gates()).backward()
    assert c.hidden_scores.grad is None and len(c.trainable_parameters())==2
    assert torch.isfinite(synchronize_controller(c))
    selection=g.full();selection['hidden_indices']=list(range(6))
    with pytest.raises(ValueError):c.set_selection(selection)

def test_all_one_hard_masks_have_exact_open_values():
    g,args,m,data=small();branch=install(m).eval();c=Controller(g)
    with torch.no_grad():a=branch(*data);b=branch(*data,*c.gates(mode='binary'))
    assert torch.equal(a['video'],b['video'])
    for x,y in zip(a['cache'],b['cache']):
        for k in x:assert torch.equal(x[k],y[k])


def test_export_roundtrip_requires_no_masks_or_teacher(tmp_path):
    from vpp2.adaptive_mask.export import publish,load_export
    g,args,m,data=small();bundle=tmp_path/'teacher';bundle.mkdir();torch.save({'dit':m.state_dict()},bundle/'video.pt')
    selection=g.full();selection['hidden_indices']=[0,2,4,6,8,10];selection['ffn_indices'][0]=list(range(8));selection['cross_head_indices'][1]=[1]
    record=publish(dict(root=str(tmp_path/'experiment'),teacher_bundle=str(bundle)),selection,g,args,'unit-test')
    assert record['auxiliary_modules_required'] is False
    compact=load_export(tmp_path/'experiment/export',device='cpu',dtype=torch.float32).eval();install_blocks(compact,())
    masked=install(m).eval();c=Controller(g);c.set_selection(selection)
    with torch.no_grad():a=masked(*data,*c.gates(mode='binary'));b=native_outputs(compact,data)
    torch.testing.assert_close(a['video'],b['video'],rtol=2e-5,atol=3e-6)
    manifest=tmp_path/'experiment/export/manifest.json'
    import json
    content=json.loads(manifest.read_text());content['files'][0]['sha256']='wrong';manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError):load_export(manifest.parent)


def test_protected_hidden_fast_path_keeps_outputs_and_gradients():
    g,args,m,data=small();branch=install(m).eval().requires_grad_(False)
    c=Controller(g,learn_hidden=False)
    with torch.no_grad():
        a=branch(*data,*c.gates(mode='binary'));b=branch(*data,*c.forward_gates(mode='binary'))
    assert torch.equal(a['video'],b['video'])
    for left,right in zip(a['cache'],b['cache']):
        for key in left:assert torch.equal(left[key],right[key])
    out=branch(*data,*c.forward_gates())
    out['video'].square().mean().backward()
    assert c.hidden_scores.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in c.trainable_parameters())


def test_late_cache_optimization_keeps_exact_terms_and_late_gradients():
    from vpp2.compact_kv.losses import kv_terms
    from vpp2.adaptive_mask.losses import cache_terms
    torch.manual_seed(7)
    left=[{k:torch.randn(2,3,40,requires_grad=True) for k in ('k','v')} for _ in range(40)]
    right=[{k:torch.randn(2,3,40) for k in ('k','v')} for _ in range(40)]
    fast=cache_terms(left,right);reference=kv_terms(left,right)
    torch.testing.assert_close(fast,reference,rtol=0,atol=0)
    params=[p for layer in left for p in layer.values()]
    a=torch.autograd.grad(fast[:,20:].sum(),params,allow_unused=True,retain_graph=True)
    b=torch.autograd.grad(reference[:,20:].sum(),params)
    assert all(g is None for g in a[:40])
    assert all(g.count_nonzero()==0 for g in b[:40])
    for x,y in zip(a[40:],b[40:]):torch.testing.assert_close(x,y,rtol=0,atol=0)
