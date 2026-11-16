import json
import torch
import pytest
from vpp2.flexible_mask.recovery_handoff import RecoveryMemoryCache,require_selected

def test_bounded_cache_is_lossless_detached_and_isolated():
    cache=RecoveryMemoryCache(8)
    x=torch.randn(2,3,dtype=torch.bfloat16,requires_grad=True)
    cache.save(1,{'x':x,'input_latents':torch.ones(1)})
    a=cache.load(1)
    assert torch.equal(a['x'],x.detach()) and not a['x'].requires_grad and a['input_latents'] is None
    a['x'].zero_();assert torch.equal(cache.load(1)['x'],x.detach())
    for i in range(2,11):cache.save(i,{'x':x})
    assert len(cache.entries)==8 and cache.load(1) is None

def test_recovery_rejects_mask_continuation(tmp_path):
    base={'root':str(tmp_path),'recovery_handoff':{'schema':'milestone-selection-recovery-v1','early_recovery_authorized':True}}
    for key in ['resume_from','pressure_continuation']:
        with pytest.raises(ValueError,match='cannot continue'):require_selected(dict(base,**{key:{'old':True}}),'unused')

def test_absent_authorization_rejected(tmp_path):
    with pytest.raises(ValueError,match='authorization'):require_selected({'root':str(tmp_path)},'unused')
