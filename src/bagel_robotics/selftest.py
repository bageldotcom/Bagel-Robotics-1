"""CPU toy-model scientific contracts; no model assets, dataset or GPU required."""

import copy

import math


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

def test_relative_reliability_and_shared_component_veto():
    t=torch.tensor([1.,2.,0.,0.,1.],requires_grad=True)
    s=torch.tensor([2.,1.,0.,2.,2.],requires_grad=True);zero=torch.ones(5)*10
    tc=torch.zeros(5,3);zc=torch.ones(5,3)*10;tc[-1,2]=20
    r,_=reliability(t,s,zero,tc,zc)
    torch.testing.assert_close(r,torch.tensor([.5,0.,0.,1.,0.]));assert not r.requires_grad


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

def main():
    torch.set_num_threads(1)
    tests=[value for name,value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"Bagel scientific CPU contracts passed: {len(tests)}")

if __name__ == "__main__":
    main()
