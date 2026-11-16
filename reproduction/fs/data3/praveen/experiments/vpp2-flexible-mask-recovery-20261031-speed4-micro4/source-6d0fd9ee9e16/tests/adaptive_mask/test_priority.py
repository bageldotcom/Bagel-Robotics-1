import copy
import torch
import pytest
from vpp2.adaptive_mask.controller import Controller
from vpp2.adaptive_mask.geometry import Geometry
from vpp2.adaptive_mask.priority import MaskPriority


def setup():
    g=Geometry(hidden=8,ffn=20,layers=3,heads=4,head_dim=2,pack=2,hidden_floor=8,ffn_floor=2,head_floor=1)
    c=Controller(g,learn_hidden=False)
    with torch.no_grad():
        c.ffn_scores.copy_(torch.linspace(1,-1,20).expand(2,-1))
        c.cross_scores.copy_(torch.linspace(1,-1,4).expand(2,-1))
    s=g.full();s['ffn_indices']=[list(range(10)),list(range(10)),list(range(20))]
    s['cross_head_indices']=[[0,1],[0,1],[0,1,2,3]];c.set_selection(s)
    return c,MaskPriority(c)


def test_priority_requires_real_decisions_history_and_boundary_separation():
    c,p=setup()
    for _ in range(4):
        p.observe();assert all((s==1).all() for s in p.scales.values())
    p.observe()
    assert p.scales['ffn_scores'][0,0]==.1 and p.scales['ffn_scores'][0,-1]==.1
    assert p.scales['ffn_scores'][0,9]==1 and p.scales['ffn_scores'][0,10]==1
    p.observe(pruning_active=False);assert len(p.history)==0
    c.set_selection(c.geometry.full())
    for _ in range(5):p.observe()
    assert all((s==1).all() for s in p.scales.values()) # all-kept is not evidence of certainty


def test_flip_and_rank_drift_reopen_scores_without_deletion():
    c,p=setup()
    for _ in range(5):p.observe()
    selection=c.selection();selection['ffn_indices'][0]=list(range(1,11));c.set_selection(selection)
    p.refresh();assert p.scales['ffn_scores'][0,0]==1
    selection['ffn_indices'][0]=list(range(10));c.set_selection(selection);p.refresh()
    p.observe();assert p.scales['ffn_scores'][0,0]==1 # flip-back between checkpoints still detected
    p.reopen();assert all((s==1).all() for s in p.scales.values())
    for _ in range(4):p.observe()
    with torch.no_grad():c.ffn_scores[0,0]-=.5
    p.observe();assert p.scales['ffn_scores'][0,0]==1
    assert c.ffn_scores.numel()==40 and all(v.numel()==40 for v in [p.scales['ffn_scores']])


def test_actual_adam_displacement_scales_and_exact_resume():
    c,p=setup();ref=copy.deepcopy(c)
    optimizer=torch.optim.AdamW(c.trainable_parameters(),lr=.01,weight_decay=0)
    original=torch.optim.AdamW(ref.trainable_parameters(),lr=.01,weight_decay=0)
    for _ in range(5):p.observe()
    before=c.ffn_scores.detach().clone()
    for parameter in c.trainable_parameters():parameter.grad=torch.ones_like(parameter)
    for parameter in ref.trainable_parameters():parameter.grad=torch.ones_like(parameter)
    p.step(optimizer);original.step()
    actual=c.ffn_scores-before;baseline=ref.ffn_scores-before
    torch.testing.assert_close(actual,baseline*p.scales['ffn_scores'],atol=1e-7,rtol=1e-5)
    saved=(copy.deepcopy(c.state_dict()),copy.deepcopy(optimizer.state_dict()),p.state_dict())
    def update():
        for parameter in c.trainable_parameters():parameter.grad=torch.ones_like(parameter)*.3
        p.step(optimizer);p.refresh();p.observe()
        return copy.deepcopy(c.state_dict()),p.state_dict()
    expected,expected_priority=update()
    c.load_state_dict(saved[0]);optimizer.load_state_dict(saved[1]);p.load_state_dict(saved[2]);actual,actual_priority=update()
    for key in ('ffn_scores','cross_scores'):assert torch.equal(actual[key],expected[key])
    for key in p.names:assert torch.equal(actual_priority['scales'][key],expected_priority['scales'][key])
    assert c.hidden_scores.grad is None
