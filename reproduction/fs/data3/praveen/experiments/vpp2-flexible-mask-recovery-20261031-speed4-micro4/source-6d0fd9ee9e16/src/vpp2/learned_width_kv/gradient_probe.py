"""Real-model action-only backward proof; KV loss cannot hide a detached action path."""
import torch
from deepspeed.utils import safe_get_full_grad

from vpp2.compact_kv.losses import action_error
from vpp2.compact_kv.model import action_prediction


def observe_cache_gradients(cache):
    if len(cache) != 40:
        raise ValueError('The action probe requires all 40 KV interfaces')
    norms = torch.zeros(40, 2, device=cache[0]['k'].device, dtype=torch.float32)
    handles = []
    for index, layer in enumerate(cache):
        for column, key in enumerate(('k', 'v')):
            value = layer[key]
            if not value.requires_grad:
                raise RuntimeError(f'Detached student cache at layer {index}, {key}')
            def record(gradient, i=index, j=column):
                norms[i, j] = gradient.detach().float().norm()
            handles.append(value.register_hook(record))
    return norms, handles


def action_only_backward(engine, controller, reference, inputs, noise, tv, noisy, ta, target, mask):
    engine.train()
    reference.mot.training = True
    engine.set_gradient_accumulation_boundary(True)
    cache = engine(inputs, noise, tv, mask[:1950, :1950], *controller.gates(100))
    norms, handles = observe_cache_gradients(cache)
    prediction = action_prediction(reference, inputs, cache, noisy, ta, mask)
    valid = inputs.get('action_loss_is_pad', inputs.get('action_is_pad'))
    loss = (action_error(prediction, target, valid) *
            reference.train_action_scheduler.training_weight(ta)).mean()
    # No teacher imitation, KV loss, or mask regularizer in this probe.
    engine.backward(loss, scale_wrt_gas=False)
    for handle in handles:
        handle.remove()
    if not torch.isfinite(norms).all() or not (norms > 0).all():
        raise RuntimeError('Ground-truth action loss did not reach every video K/V interface')
    hidden = controller.hidden_scores.grad
    ffn = controller.ffn_scores.grad
    if hidden is None or ffn is None or not torch.isfinite(hidden).all() or not torch.isfinite(ffn).all():
        raise RuntimeError('Missing or nonfinite action-only channel-score gradients')
    if not hidden.abs().sum() > 0 or not (ffn.abs().sum(-1) > 0).all():
        raise RuntimeError('Action supervision did not reach hidden and every learned FFN mask')
    patch_gradient = safe_get_full_grad(engine.module.video.patch_embedding.bias)
    if patch_gradient is None or not torch.isfinite(patch_gradient).all() or not patch_gradient.abs().sum() > 0:
        raise RuntimeError('Action supervision did not reach native video parameters')
    if any(p.grad is not None for p in reference.parameters()):
        raise RuntimeError('Frozen teacher/action parameters acquired gradients')
    result = dict(action_only=True, kv_loss_used=False, optimizer_step_taken=False,
                  cache_gradient_norms=norms.cpu().tolist(),
                  hidden_score_gradient_norm=float(hidden.norm()),
                  ffn_score_gradient_norms=ffn.norm(dim=-1).cpu().tolist(),
                  patch_bias_gradient_norm=float(patch_gradient.norm()),
                  frozen_reference_has_gradients=False)
    # This integration process is disposable; no parameters or optimizer state
    # from it are used to initialize either phase of training.
    return result
