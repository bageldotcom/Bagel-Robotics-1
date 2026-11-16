from types import SimpleNamespace
import pytest
import torch
from vpp2.critical_windows.inference import masked_prefill


class FakeModel:
    def _build_frozen_video_cache(self, **kwargs):
        return 'native'
    def _build_mot_attention_mask(self,*args):
        return torch.ones((1982,1982),dtype=torch.bool)


def test_mask_override_is_scoped_and_restored_even_on_error():
    model=FakeModel();calls=[]
    def branch(*args):calls.append(args);return ['cache']
    args=dict(latents=torch.zeros(1,16,5,30,52),timestep_video=torch.tensor([1000]),
              context=None,context_mask=None,condition_latents=None,clip_fea=None,action_seq_len=32)
    with masked_prefill(model,branch):
        assert model._build_frozen_video_cache(**args)[0]==['cache']
        assert len(calls)==1
    assert model._build_frozen_video_cache()=='native'
    assert '_build_frozen_video_cache' not in model.__dict__
    with pytest.raises(ValueError):
        with masked_prefill(model,branch):model._build_frozen_video_cache(**dict(args,return_video_prediction=True))
    assert model._build_frozen_video_cache()=='native'


def test_mask_route_rejects_incompatible_video_dimensions():
    model=FakeModel()
    with masked_prefill(model,lambda *a: []),pytest.raises(ValueError,match='dimensions'):
        model._build_frozen_video_cache(latents=torch.zeros(1,16,1,30,52),timestep_video=None,
            context=None,context_mask=None,condition_latents=None,clip_fea=None,action_seq_len=32)
