import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from vpp2.flexible_mask.losses import objectives
from vpp2.flexible_mask.replay import initial_coefficient,validate
from vpp2.flexible_mask.pressure import weight
from vpp2.flexible_mask.runtime import Runtime
from vpp2.flexible_mask.gradient_probe import frozen_weight_mask_gradient

BASE=0.011596363996894921

def configuration():
    c=json.loads((Path(__file__).resolve().parents[2]/'configs/flexible_mask/full30k.json').read_text())
    c['train']['kv_weight']=0.
    c['regularization_replay']={'schema':'reference-regularization-replay-v1','reference_root':'/reference','reference_config_sha256':'a'*64,
      'resolved_update':200,'base_coefficient':BASE,'pressure_start':1251,'pressure_ramp':250,'pressure_multiplier':2.,'final_coefficient':2*BASE}
    return c


def test_replayed_weights_match_original_schedule_at_every_update():
    c=configuration();k=initial_coefficient(c)
    ref=copy.deepcopy(c);ref.pop('regularization_replay')
    ref['pressure_continuation']={'schema':'vpp2-size-pressure-ramp-v1','start_update':1251,'ramp_updates':250,'target_multiplier':2.,
       'base_coefficient':BASE,'absolute_cap':2*BASE,'target_video_parameters':10_000_000_000,'automatic_second_increase':False}
    for u in range(3001):assert weight(k,c,u)==weight(k,ref,u)
    assert not any(k.collecting(u) for u in range(3001))
    assert weight(k,c,0)==0 and weight(k,c,1501)==2*BASE


def test_no_old_training_state_can_seed_fresh_ablation():
    c=configuration();c['resume_from']={'root':'old'}
    with pytest.raises(ValueError,match='cannot resume'):validate(c)


def test_no_kv_loss_is_exactly_gt_plus_reliable_video():
    student=torch.tensor([[[2.]],[[1.]]],requires_grad=True)
    teacher=torch.tensor([[[1.]],[[2.]]]);target=torch.ones_like(student)*3
    video=torch.ones(2,1,5,2,2,requires_grad=True);tv=torch.zeros_like(video)
    loss,metrics,details,terms=objectives(student,teacher,target,None,torch.ones(2),video,tv,None,torch.ones(2,2),1.,kv_weight=0.)
    assert torch.equal(loss,terms['gt']+.05*terms['video'])
    assert terms['kv'].item()==0 and not terms['kv'].requires_grad
    assert metrics['late_kv_weight']==0 and not metrics['kv_comparison_computed']
    loss.backward();assert student.grad.abs().sum()>0 and video.grad.abs().sum()>0
    assert video.grad[0].abs().sum()==0


def test_cache_comparison_is_never_called_when_disabled(monkeypatch):
    from vpp2.flexible_mask import runtime
    def fail(*a):raise AssertionError('K/V comparison must be skipped')
    monkeypatch.setattr(runtime,'cache_terms',fail)
    assert Runtime.kv_terms(SimpleNamespace(config={'train':{'kv_weight':0.}}),object(),object()) is None


def test_zero_weight_does_not_silently_compute_cache_comparison():
    x=torch.zeros(1,1,1)
    with pytest.raises(ValueError,match='must be skipped'):
        objectives(x,x,x,None,torch.ones(1),None,None,torch.zeros(1,40,2),None,1.,kv_weight=0.)


def test_gradient_probe_accepts_only_explicit_disabled_kv():
    module=torch.nn.Linear(2,2);mask=torch.nn.Parameter(torch.ones(2))
    def objective(y):return {'gt':(y*mask).square().mean(),'video':(y*mask-1).square().mean(),'kv':y.detach().new_zeros(())}
    result=frozen_weight_mask_gradient(module,(mask,),objective,(torch.ones(1,2),))
    assert result['gt'][0].abs().sum()>0 and result['kv'][0].abs().sum()==0
    assert mask.grad is None and all(p.grad is None for p in module.parameters())


def test_teacher_disagreement_is_reported_without_changing_gt_ranking():
    from vpp2.flexible_mask.selection import aggregate
    r=[dict(window_id=i,task='a',suite='precision',student_error=2.,teacher_error=1.,student_teacher_error=float(i)) for i in range(4)]
    a=aggregate(r);assert a['task/a']['student_teacher_error']==1.5
    assert a['task/a']['student_error']==2.
