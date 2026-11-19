from types import SimpleNamespace
import pytest
import torch
from bagel_robotics.recovery.losses import objectives,reliability


def tensors():
    s=torch.ones(2,4,16,requires_grad=True);t=torch.zeros_like(s);target=torch.zeros_like(s)
    sv=torch.ones(2,2,5,2,2,requires_grad=True);tv=torch.zeros_like(sv)
    valid=torch.ones(2,2,dtype=torch.bool)
    return s,t,target,sv,tv,valid

def test_no_kv_matches_exact_two_loss_formula_and_has_no_cache_gradient():
    s,t,target,sv,tv,valid=tensors()
    loss,m,d,terms=objectives(s,t,target,None,torch.ones(2),sv,tv,None,valid,1.,kv_weight=0.)
    assert torch.equal(loss,terms['gt']+.05*terms['video'])
    assert m['kv_comparisons_measured']==0 and m['effective_kv_weight']==0
    assert 'late_kv_raw' not in m and not any(k.startswith('kv/layer_') for k in m)
    assert terms['kv'].requires_grad is False and terms['kv'].item()==0
    assert d['reliability'].requires_grad is False
    loss.backward()
    assert s.grad.abs().sum()>0 and sv.grad.abs().sum()>0

def test_missing_kv_fails_if_loss_is_active():
    s,t,target,sv,tv,valid=tensors()
    with pytest.raises(ValueError,match='active loss'):
        objectives(s,t,target,None,torch.ones(2),sv,tv,None,valid,1.,kv_weight=.05)

def test_unreliable_teacher_cannot_teach_video_but_gt_remains_active():
    s,t,target,sv,tv,valid=tensors();t=t+100.
    loss,m,d,terms=objectives(s,t,target,None,torch.ones(2),sv,tv,None,valid,1.,kv_weight=0.)
    assert torch.equal(d['reliability'],torch.zeros(2))
    assert torch.equal(loss,terms['gt']) and terms['video'].item()==0
    loss.backward();assert s.grad.abs().sum()>0 and sv.grad.abs().sum()==0

def test_no_valid_future_leaves_gt_action_training():
    s,t,target,sv,tv,valid=tensors();valid.zero_()
    loss,m,d,terms=objectives(s,t,target,None,torch.ones(2),sv,tv,None,valid,1.,kv_weight=0.)
    assert torch.equal(loss,terms['gt'])

def test_runtime_never_calls_cache_comparison_when_disabled(monkeypatch):
    import bagel_robotics.adaptive_pruning.runtime as module
    rt=object.__new__(module.Runtime)
    rt.config={'seed':100008,'execution':{},'train':{'action_gt_weight':1.,'teacher_video_weight':.05,'kv_weight':0.}}
    rt.engine=SimpleNamespace(_scale_wrt_gas=True);rt.reader=None;rt.device=torch.device('cpu');rt.masks=None
    rt.reference=SimpleNamespace(train_action_scheduler=SimpleNamespace(training_weight=lambda t:torch.ones(len(t))))
    rt.gates=lambda *a:None;calls=[]
    def prepare(ref,reader,rows,positions,seed,strata):
        b=len(rows)
        return ({'count':b,'action_loss_is_pad':None},None,None,None,torch.ones(b),torch.zeros(b,4,16),None)
    def video(parts,teacher=False,**kwargs):
        calls.append('teacher_video' if teacher else 'student_video')
        return {'cache':0. if teacher else 1.,'video':torch.ones(parts[0]['count'],2,5,2,2,requires_grad=not teacher)}
    def predict(parts,cache):
        calls.append('action')
        return torch.full((parts[0]['count'],4,16),cache,requires_grad=bool(cache))
    rt.video=video;rt.predict=predict
    monkeypatch.setattr(module,'prepare',prepare)
    monkeypatch.setattr(module,'future_mask',lambda rows,device:torch.ones(len(rows),2))
    def forbidden(*a):raise AssertionError('Disabled K/V comparison executed')
    monkeypatch.setattr(module,'cache_terms',forbidden)
    loss,m,_,_=rt.forward([0,1],[100,101],56)
    assert calls==['teacher_video','action','student_video','action']
    assert m['kv_comparisons_measured']==0
    loss.backward()

def test_reliability_rule_not_changed():
    s=torch.tensor([2.,4.],requires_grad=True);t=torch.tensor([1.,1.])
    r,_=reliability(t,s,torch.tensor([10.,10.]),torch.ones(2,3),torch.full((2,3),10.))
    assert not r.requires_grad and torch.equal(r,torch.tensor([.5,.75]))

def test_original_loss_arithmetic_unchanged():
    s,t,target,sv,tv,valid=tensors();kv=torch.ones(2,40,2,requires_grad=True)
    loss,_,_,terms=objectives(s,t,target,None,torch.ones(2),sv,tv,kv,valid,1.)
    assert torch.equal(loss,terms['gt']+.05*(terms['video']+terms['kv']))
