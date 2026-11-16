from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from vpp2.flexible_mask.recovery_gradients import sample_indices,sample_parameters,sampled_update,scalar_summary


@pytest.mark.parametrize('length',[0,1,2,7,63,64,65,1024,16_777_220,18_677_760,33_554_432,2**31+1])
def test_diagnostic_indices_stay_in_bounds(length):
    x=sample_indices(length)
    assert x.dtype==torch.int64 and len(x)==min(length,64)
    if length:
        assert x.min()==0 and x.max()<length and len(x.unique())==len(x)
        if length>1:assert x[-1]==length-1


def test_original_float_endpoint_failure_is_fixed():
    n=18_677_760
    assert torch.linspace(0,n-1,64,dtype=torch.float32).long()[-1]==n
    assert sample_indices(n)[-1]==n-1


def test_indices_are_independent_of_default_float_dtype():
    old=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        x=sample_indices(18_677_760)
        assert x[-1]==18_677_759 and x.dtype==torch.int64
    finally:torch.set_default_dtype(old)


def test_sampled_updates_are_read_only_and_handle_empty_shards():
    model=torch.nn.Module();model.register_parameter('empty',torch.nn.Parameter(torch.empty(0)))
    model.register_parameter('weight',torch.nn.Parameter(torch.ones(100)))
    opt=SimpleNamespace(get_local_fp32_param=lambda p:p)
    rt=SimpleNamespace(engine=SimpleNamespace(module=model,optimizer=opt),device='cpu')
    before=sample_parameters(rt)
    assert 'empty' not in before and torch.equal(model.weight,torch.ones(100))
    with torch.no_grad():model.weight.add_(.25)
    with patch('vpp2.flexible_mask.recovery_gradients.dist.all_reduce',lambda t,op:None):d=sampled_update(rt,before)
    assert d['optimizer_update_sampled/all/rms_estimate']==pytest.approx(.25)
    assert d['optimizer_update_sampled/all/relative_l2_estimate']==pytest.approx(.25)
    assert torch.equal(model.weight,torch.full((100,),1.25))


def test_loss_ranges_are_per_window_and_keep_reliability_weighting_visible():
    keys=('gt_action_loss','teacher_video_raw','teacher_video_gated','weighted_gt_action_loss','weighted_teacher_video_loss','reliability')
    rows=[dict(zip(keys,[1.,10.,0.,1.,0.,0.])),dict(zip(keys,[3.,10.,5.,3.,.25,.5]))]
    s=scalar_summary(rows)
    assert s['samples']==2
    assert s['gt_action_loss']['mean']==2
    assert s['teacher_video_raw']['mean']==10
    assert s['weighted_teacher_video_loss']==dict(min=0.,p50=.125,p95=.2375,max=.25,mean=.125)
