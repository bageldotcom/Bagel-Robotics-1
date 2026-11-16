import copy
import json
from collections import Counter

import pytest
import torch

from vpp2.adaptive_mask.hidden import HiddenSearch
from vpp2.adaptive_mask.controller import Controller
from vpp2.adaptive_mask.geometry import Geometry, ParameterInventory
from vpp2.adaptive_mask.priority import MaskPriority
from vpp2.adaptive_mask.budget import Budget
from vpp2.adaptive_mask.feedback import selection_churn
from vpp2.adaptive_mask.monitor import component_quality, quality_flags
from vpp2.adaptive_mask.state import snapshot, restore
from vpp2.adaptive_mask.masking import install
from vpp2.adaptive_mask.export import build_compact, sliced_state
from vpp2.models.wan21_14b.wan_video_dit import WanVideoDiT


GOOD = dict(passed=True, clearly_passing=True, resolved_violation=False, complete=True, source='fitting_only')
CHURN = dict(hidden=0., ffn_max=0., cross_heads_max=0.)


def setup():
    geometry = Geometry(hidden=80, ffn=128, layers=3, heads=4, head_dim=8,
                        pack=2, hidden_floor=76, ffn_floor=16, head_floor=1)
    args = dict(hidden_dim=80, ffn_dim=128, num_layers=3, num_heads=4,
                attn_head_dim=8, freq_dim=8, text_dim=10, in_dim=8, out_dim=2, cross_attn_norm=True)
    with torch.device('meta'):
        model = WanVideoDiT(**args)
    inventory = ParameterInventory.from_model(model, geometry)
    controller = Controller(geometry)
    with torch.no_grad():
        controller.hidden_scores.copy_(torch.linspace(1, -1, 80))
    budget = Budget(original=inventory.original, target=int(inventory.original*.7),
                    milestones=[[560,int(inventory.original*.9)],[1120,int(inventory.original*.8)],[1680,int(inventory.original*.7)]])
    budget.attempt = budget.progress = 280
    search = HiddenSearch(widths=[80,78,76], width=80)
    for _ in range(3):
        search.observe(controller)
    return controller, inventory, budget, search, args


def start(controller, inventory, budget, search):
    return search.start(budget.attempt, controller, inventory, budget, GOOD, CHURN, True)


def test_hidden_trial_requires_quality_and_stable_rankings():
    c, inv, b, h, _ = setup()
    assert not h.start(280,c,inv,b,dict(GOOD,passed=False),CHURN,True)
    assert not h.start(280,c,inv,b,GOOD,CHURN,False)
    assert not h.start(280,c,inv,b,GOOD,dict(CHURN,hidden=.01),True)
    h.candidates[-1]['indices'] = list(range(2,80))
    assert not start(c,inv,b,h)
    h.candidates.clear()
    for _ in range(3):h.observe(c)
    assert start(c,inv,b,h) and h.width==78 and h.active


def test_trial_isolates_hidden_removal_and_holds_capacity():
    c, inv, b, h, _ = setup();anchor=c.selection()
    assert start(c,inv,b,h)
    original_cap=b.cap
    with torch.no_grad():
        c.ffn_scores.normal_();c.cross_scores.normal_();c.hidden_scores[-1]=10
    h.project(c,inv,b.cap)
    assert c.selection()['ffn_indices']==anchor['ffn_indices']
    assert c.selection()['cross_head_indices']==anchor['cross_head_indices']
    assert 79 in c.selection()['hidden_indices']
    assert inv.count(c.selection()) < inv.count(anchor)
    for _ in range(70):b.advance(hold_capacity=h.active)
    assert b.cap==original_cap
    assert b.observe_fitting(GOOD,CHURN,hold_capacity=h.active)=='hold_isolated_hidden_trial'
    with pytest.raises(ValueError):h.project(c,inv,b.cap-1)


def test_trial_acceptance_needs_full_check_and_uncertainty_is_not_success():
    c,inv,b,h,_=setup();start(c,inv,b,h)
    assert h.assess(350,GOOD,GOOD,False,CHURN)=='refine_isolated_hidden_trial'
    assert h.assess(560,GOOD,GOOD,True,CHURN)=='accept_hidden_width'
    h.accept();assert h.width==78 and not h.active
    for _ in range(3):h.observe(c)
    b.attempt=560
    assert start(c,inv,b,h) and h.width==76
    assert h.assess(840,dict(GOOD,passed=False),GOOD,True,CHURN)=='reject_hidden_unresolved_quality_or_turnover'
    with pytest.raises(ValueError):h.accept()


def test_confirmed_harm_restores_complete_training_state_and_blocks_deeper_trial():
    c,inv,b,h,_=setup();o=torch.optim.AdamW(c.trainable_parameters(),lr=.01,weight_decay=0)
    priority=MaskPriority(c);parent=snapshot(c,o,256,1,Counter({1:2}),0);saved_priority=priority.state_dict()
    assert start(c,inv,b,h)
    sum(g.sum() for g in c.gates()).backward();priority.step(o)
    assert h.assess(350,dict(GOOD,passed=False,resolved_violation=True),GOOD,False,CHURN)=='reject_hidden_confirmed_harm'
    cursor,lineage,coverage,ramp=restore(parent,c,o);priority.load_state_dict(saved_priority);priority.reopen()
    h.restored(c.selection())
    assert (cursor,lineage,coverage,ramp)==(256,1,Counter({1:2}),0)
    assert c.selection()==parent['controller']['_extra_state'] and h.width==80 and h.blocked
    for k in ('hidden_scores','ffn_scores','cross_scores'):assert torch.equal(c.state_dict()[k],parent['controller'][k])
    assert not start(c,inv,b,h)


def test_hidden_priority_scales_actual_updates_and_keeps_off_channels_trainable():
    c,inv,b,h,_=setup();start(c,inv,b,h)
    p=MaskPriority(c,boundary_fraction=.001);o=torch.optim.AdamW(c.trainable_parameters(),lr=.01,weight_decay=0)
    for _ in range(5):p.observe()
    assert 'hidden_scores' in p.names
    assert p.scales['hidden_scores'][0]==.1 and p.scales['hidden_scores'][-1]==.1
    loss=sum(g.sum() for g in c.gates());loss.backward()
    assert c.hidden_scores.grad[-1]!=0
    p.step(o);p.reopen()
    assert (p.scales['hidden_scores']==1).all()


def test_resume_in_middle_of_hidden_trial_reproduces_masks_optimizer_rng_and_priority():
    c,inv,b,h,_=setup();start(c,inv,b,h)
    o=torch.optim.AdamW(c.trainable_parameters(),lr=.01,weight_decay=0);p=MaskPriority(c)
    def update():
        h.project(c,inv,b.cap);o.zero_grad(set_to_none=True)
        sum((g*torch.randn_like(g)).sum() for g in c.gates()).backward();p.step(o)
        h.project(c,inv,b.cap);p.refresh();b.advance(hold_capacity=h.active)
    update();saved=snapshot(c,o,512,2,Counter({1:2,2:2}),1)
    saved_h=json.loads(json.dumps(h.state_dict()));saved_b=b.state_dict();saved_p=p.state_dict()
    update();expected=copy.deepcopy(c.state_dict());expected_o=copy.deepcopy(o.state_dict())
    expected_b=b.state_dict();expected_p=p.state_dict()
    restore(saved,c,o);h=HiddenSearch(**saved_h);b=Budget(**saved_b);p.load_state_dict(saved_p);update()
    assert b.state_dict()==expected_b and c.selection()==expected['_extra_state']
    for k in ('hidden_scores','ffn_scores','cross_scores'):assert torch.equal(c.state_dict()[k],expected[k])
    for k,v in expected_o['state'].items():
        for name,value in v.items():assert torch.equal(o.state_dict()['state'][k][name],value)
    for k in p.names:assert torch.equal(p.scales[k],expected_p['scales'][k])


@pytest.mark.parametrize('width',[78,76])
def test_moderate_noncontiguous_hidden_pruning_matches_physical_model(width):
    c,inv,b,h,args=setup();torch.manual_seed(24)
    model=WanVideoDiT(**args).float().eval().requires_grad_(False)
    selection=c.selection();selection['hidden_indices']=sorted(torch.randperm(80)[:width].tolist())
    selection['ffn_indices'][0]=list(range(0,128,2));selection['cross_head_indices'][1]=[0,2,3]
    c.set_selection(selection);branch=install(model).eval()
    compact=build_compact(args,selection,c.geometry,device='cpu').float().eval()
    values,_=sliced_state(model.state_dict(),compact,selection,c.geometry);compact.load_state_dict(values,strict=True)
    compact_branch=install(compact).eval()
    inputs=dict(context=torch.randn(1,3,10),context_mask=torch.ones(1,3,dtype=torch.bool),
                clip_fea=torch.randn(1,257,1280),condition_latents=torch.randn(1,2,1,4,4))
    data=(inputs,torch.randn(1,2,2,4,4),torch.tensor([1000.]),torch.ones(8,8,dtype=torch.bool))
    with torch.no_grad():a=branch(*data,*c.forward_gates(mode='binary'));z=compact_branch(*data)
    torch.testing.assert_close(a['video'],z['video'],rtol=2e-5,atol=3e-6)
    for x,y in zip(a['cache'],z['cache']):
        for k in ('k','v'):
            assert x[k].shape[-1]==32 # original action-facing width, independent of 80/78/76
            torch.testing.assert_close(x[k],y[k],rtol=2e-5,atol=3e-6)
    assert inv.count(selection)==sum(p.numel() for p in compact.parameters())


def test_component_regression_cannot_hide_behind_equal_aggregate_action_error():
    rows=[dict(task='precision',suite='precision',episode=i,frame=0,teacher_error=1.,student_error=1.,
               teacher_components=[1.,1.,1.],student_components=[.7,1.,1.3]) for i in range(8)]
    result=component_quality(rows,None,{})
    assert result['gripper']['resolved_violation']
    assert not quality_flags(list(result.values()))['passed']


def test_parent_restore_is_required_even_when_bad_identities_have_same_capacity():
    b=Budget(original=16_394_878_784);b.attempt=280
    assert b.cap==b.accepted_cap
    assert b.decide(False,resolved_violation=True)=='rescue_confirmed_harm_once'
    for _ in range(280):b.advance()
    assert b.decide(False,resolved_violation=True)=='restore_parent_and_stop_shrinking'


def test_extra_hidden_quality_check_never_ends_search_early():
    b=Budget(original=16_394_878_784);b.attempt=840
    assert b.decide(True,stable=True,coverage=True,extra_quality_check=True)=='accept'
    with pytest.raises(ValueError):b.decide(True)
