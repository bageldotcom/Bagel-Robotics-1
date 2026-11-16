"""Replicated FP32 pruning scores; never register this module with DeepSpeed."""
import math
import torch
from torch import nn
import torch.distributed as dist


def stable_topk(scores, count):
    if not 0 < count <= scores.numel() or not torch.isfinite(scores).all():
        raise ValueError('Invalid scores or channel budget')
    # Stable descending sort resolves ties by the original channel index.
    return torch.argsort(scores.detach(), descending=True, stable=True)[:count].sort().values


def schedule(update):
    if not 0 <= update <= 1000:
        raise ValueError('Mask schedule is defined on optimizer updates 0..1000')
    return dict(alpha=min(update / 100., 1.),
                temperature=1. + 9. * min(max((update - 100) / 700., 0.), 1.),
                discrete_ramp=min(max((update - 200) / 600., 0.), 1.))


class MaskController(nn.Module):
    def __init__(self, selection, hidden=5120, ffn=13824, layers=40, keep_hidden=2048, keep_ffn=8192):
        super().__init__()
        if len(selection['ffn_indices']) != layers:
            raise ValueError('Every original layer needs a final FFN selection')
        self.hidden, self.ffn, self.layers = hidden, ffn, layers
        self.keep_hidden, self.keep_ffn = keep_hidden, keep_ffn
        self.final_ffn = list(selection['ffn_indices'][-1])
        def scores(size, indices):
            if len(indices) != len(set(indices)) or not all(0 <= i < size for i in indices):
                raise ValueError('Invalid initialization indices')
            x = torch.full((size,), -1.5, dtype=torch.float32)
            x[indices] = 1.5
            return x
        if len(selection['hidden_indices']) != keep_hidden or any(len(v) != keep_ffn for v in selection['ffn_indices']):
            raise ValueError('Initialization does not meet the final channel budget')
        self.hidden_scores = nn.Parameter(scores(hidden, selection['hidden_indices']))
        self.ffn_scores = nn.Parameter(torch.stack([scores(ffn, v) for v in selection['ffn_indices'][:-1]]))

    def gates(self, update, mode='soft'):
        if self.hidden_scores.dtype != torch.float32 or self.ffn_scores.dtype != torch.float32:
            raise RuntimeError('Mask scores must remain FP32')
        if mode == 'open':
            return None, None
        if mode == 'binary':
            hidden = torch.zeros_like(self.hidden_scores)
            hidden[stable_topk(self.hidden_scores, self.keep_hidden)] = 1.
            ffns = torch.zeros_like(self.ffn_scores)
            for i, row in enumerate(self.ffn_scores):
                ffns[i, stable_topk(row, self.keep_ffn)] = 1.
            return hidden, ffns
        if mode != 'soft':
            raise ValueError(mode)
        s = schedule(update)
        def gate(x):
            return (1. - s['alpha']) + s['alpha'] * torch.sigmoid(s['temperature'] * x)
        return gate(self.hidden_scores), gate(self.ffn_scores)

    def regularizers(self, update, gates=None):
        h, f = self.gates(update) if gates is None else gates
        s = schedule(update)
        def penalties(g, total, keep):
            target = total + s['alpha'] * (keep - total)
            count = ((g.sum(-1) - target) / target).square().mean()
            # Clamp logarithms only: unlike casting scores to BF16, this keeps
            # gradients of both terms well defined near the endpoints.
            entropy = -(g * g.clamp_min(1e-7).log() + (1-g) * (1-g).clamp_min(1e-7).log()).mean() / math.log(2.)
            discrete = (2. * torch.minimum(g, 1-g)).mean()
            return torch.stack([count, entropy, discrete])
        values = .5 * penalties(h, self.hidden, self.keep_hidden) + .5 * penalties(f, self.ffn, self.keep_ffn)
        loss = 10. * values[0] + .01 * s['discrete_ramp'] * (values[1] + values[2])
        return loss, dict(zip(('count', 'entropy', 'discreteness'), values.unbind()))

    @torch.no_grad()
    def selection(self):
        return dict(schema='vpp2-learned-width-selection-v1', method='learned_action_kv_masks',
                    hidden_indices=stable_topk(self.hidden_scores, self.keep_hidden).cpu().tolist(),
                    ffn_indices=[stable_topk(row, self.keep_ffn).cpu().tolist() for row in self.ffn_scores] + [self.final_ffn],
                    final_ffn_learned=False, mask_parameters=sum(p.numel() for p in self.parameters()))

    @torch.no_grad()
    def diagnostics(self, update):
        h, f = self.gates(update)
        return dict(hidden_soft_count=float(h.sum()), ffn_soft_count=f.sum(-1).cpu().tolist(),
                    hidden_midpoint_fraction=float(((h>.1)&(h<.9)).float().mean()),
                    ffn_midpoint_fraction=float(((f>.1)&(f<.9)).float().mean()), **schedule(update))


def synchronize_controller(controller, max_norm=1.):
    """Called exactly once per optimizer update, after all microbatch backwards."""
    for parameter in controller.parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError('Missing or nonfinite mask gradient')
        if dist.is_initialized():
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(dist.get_world_size())
    return torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm, error_if_nonfinite=True)
