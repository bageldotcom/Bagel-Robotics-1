import pytest
import torch
from vpp2.adaptive_mask.equivalence import assess, streamed_fp32
from vpp2.adaptive_mask.controller import Controller
from vpp2.adaptive_mask.geometry import Geometry
from vpp2.adaptive_mask.export import build_compact, sliced_state
from vpp2.adaptive_mask.masking import install
from vpp2.models.wan21_14b.wan_video_dit import WanVideoDiT


def reports():
    fp = ['embedding_' + k for k in ('tokens', 't', 't_mod', 'context')] + ['video']
    fp += [f'layer_{i:02d}_{k}' for i in range(3) for k in ('k', 'v', 'hidden')]
    bf = ['video', 'action'] + [f'layer_{i:02d}_{k}' for i in range(3) for k in ('k', 'v')]
    return ([dict(tensor=n, relative_rmse=4e-6, max_absolute=4e-6) for n in fp],
            [dict(tensor=n, relative_rmse=.018 if n.startswith('layer_') else .004, max_absolute=.018) for n in bf])


def test_approved_rule_reports_cache_rounding_without_claiming_policy_retention():
    fp, bf = reports()
    result = assess(fp, bf, 3, 2e-5, .01)
    assert result['passed'] and result['bf16_cache_max_relative_rmse'] == .018
    assert not result['bf16_cache_limit_enforced'] and not result['policy_retention_established']


@pytest.mark.parametrize('failure', ['fp32', 'action', 'video', 'missing', 'duplicate', 'nan', 'inf'])
def test_rule_rejects_bad_or_incomplete_evidence(failure):
    fp, bf = reports()
    if failure == 'fp32': fp[0]['relative_rmse'] = .0001
    elif failure in ('action', 'video'):
        next(r for r in bf if r['tensor'] == failure)['relative_rmse'] = .011
    elif failure == 'missing': fp.pop()
    elif failure == 'duplicate': bf.append(dict(bf[-1]))
    else: bf[-1]['relative_rmse'] = float(failure)
    assert not assess(fp, bf, 3, 2e-5, .01)['passed']


@pytest.mark.parametrize('fail', [False, True])
def test_full_streamed_structure_and_cleanup(fail, monkeypatch):
    torch.manual_seed(24)
    geometry = Geometry(hidden=80, ffn=128, layers=3, heads=4, head_dim=8,
                        pack=2, hidden_floor=76, ffn_floor=16, head_floor=1)
    args = dict(hidden_dim=80, ffn_dim=128, num_layers=3, num_heads=4,
                attn_head_dim=8, freq_dim=8, text_dim=10, in_dim=8, out_dim=2, cross_attn_norm=True)
    controller = Controller(geometry)
    selection = geometry.full()
    selection['hidden_indices'] = sorted(torch.randperm(80)[:76].tolist())
    selection['ffn_indices'][0] = list(range(0,128,2))
    selection['cross_head_indices'][1] = [0,2,3]
    controller.set_selection(selection)
    wide = WanVideoDiT(**args).bfloat16().eval().requires_grad_(False)
    compact = build_compact(args, selection, geometry, device='cpu').bfloat16().eval().requires_grad_(False)
    states, _ = sliced_state(wide.state_dict(), compact, selection, geometry)
    compact.load_state_dict(states, strict=True)
    install(wide).eval(); install(compact).eval()
    originals = [{k: v.clone() for k, v in m.state_dict().items()} for m in (wide, compact)]
    inputs = dict(context=torch.randn(1,3,10).bfloat16(), context_mask=torch.ones(1,3,dtype=torch.bool),
                  clip_fea=torch.randn(1,257,1280).bfloat16(), condition_latents=torch.randn(1,2,1,4,4).bfloat16())
    parts = (inputs, torch.randn(1,2,2,4,4).bfloat16(), torch.tensor([1000.]), None, None, None,
             torch.ones(8,8,dtype=torch.bool))
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    if fail:
        def broken(*a, **kw): raise RuntimeError('injected comparison failure')
        monkeypatch.setattr(compact.blocks[1], 'forward', broken)
        with pytest.raises(RuntimeError, match='injected'):
            streamed_fp32(wide, compact, parts, controller)
    else:
        result = streamed_fp32(wide, compact, parts, controller)
        assert len(result) == 14 and max(r['relative_rmse'] for r in result) < 2e-5
    assert flags == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    for model, original in zip((wide, compact), originals):
        for name, tensor in model.state_dict().items():
            assert tensor.device.type == 'cpu' and tensor.dtype == original[name].dtype
            assert torch.equal(tensor, original[name])
