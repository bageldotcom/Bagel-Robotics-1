from types import SimpleNamespace
import pytest
import torch
from vpp2.adaptive_mask.execution import configure_recomputation
from vpp2.adaptive_mask.controller import Controller
from vpp2.adaptive_mask.geometry import Geometry
from vpp2.adaptive_mask.masking import install
from vpp2.models.wan21_14b.wan_video_dit import WanVideoDiT


@pytest.mark.parametrize('layers,action', [([1,0],True), ([0,0],True), ([3],True), ([True],True), ([0],1)])
def test_invalid_execution_policy_is_rejected(layers, action):
    model = SimpleNamespace(video_expert=SimpleNamespace(blocks=[0,1,2]), action_expert=SimpleNamespace())
    with pytest.raises(ValueError):
        configure_recomputation(model, {'execution':dict(video_checkpoint_layers=layers, action_checkpointing=action)})


def test_recomputation_policies_preserve_outputs_and_all_mask_gradients():
    torch.manual_seed(421)
    geometry = Geometry(hidden=80, ffn=128, layers=3, heads=4, head_dim=8,
                        pack=2, hidden_floor=76, ffn_floor=16, head_floor=1)
    video = WanVideoDiT(hidden_dim=80, ffn_dim=128, num_layers=3, num_heads=4,
                       attn_head_dim=8, freq_dim=8, text_dim=10, in_dim=8, out_dim=2, cross_attn_norm=True)
    video.requires_grad_(False); controller = Controller(geometry)
    selection = geometry.full(); selection['hidden_indices'] = sorted(torch.randperm(80)[:76].tolist())
    selection['ffn_indices'][0] = list(range(0,128,2)); selection['cross_head_indices'][1] = [0,2,3]
    controller.set_selection(selection); branch = install(video).train()
    model = SimpleNamespace(video_expert=video, action_expert=SimpleNamespace())
    inputs = dict(context=torch.randn(1,3,10), context_mask=torch.ones(1,3,dtype=torch.bool),
                  clip_fea=torch.randn(1,257,1280), condition_latents=torch.randn(1,2,1,4,4))
    data = (inputs,torch.randn(1,2,2,4,4),torch.tensor([1000.]),torch.ones(8,8,dtype=torch.bool))
    expected = None
    for layers in ([0,1,2], [0,2], []):
        configure_recomputation(model, {'execution':dict(video_checkpoint_layers=layers,action_checkpointing=False)})
        controller.zero_grad(set_to_none=True)
        result = branch(*data,*controller.forward_gates())
        loss = result['video'].float().square().mean() + sum(x[k].float().square().mean() for x in result['cache'] for k in ('k','v'))
        loss.backward()
        values = [result['video'].detach()] + [x[k].detach() for x in result['cache'] for k in ('k','v')]
        gradients = [p.grad.detach().clone() for p in controller.trainable_parameters()]
        assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in gradients)
        if expected is None: expected = (values,gradients)
        else:
            for a,b in zip(values+gradients,expected[0]+expected[1]):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert all(not p.requires_grad and p.grad is None for p in video.parameters())
