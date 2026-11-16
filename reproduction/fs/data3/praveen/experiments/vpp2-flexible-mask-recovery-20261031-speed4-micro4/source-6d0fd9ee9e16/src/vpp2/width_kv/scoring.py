"""Action-output Jacobian and demonstration-loss sensitivity, with task-balanced selection."""
import torch
from torch import nn

from .slicing import evenly_spaced


class ChannelGate(nn.Module):
    def __init__(self, width, device=None, dtype=torch.float32):
        super().__init__()
        self.multiplier = nn.Parameter(torch.ones(width, device=device, dtype=dtype))

    def forward(self, x):
        return x * self.multiplier.to(dtype=x.dtype)


def install_gates(video):
    """Hooks are local to this calibration copy. All-one gates preserve its function."""
    residual = nn.ModuleList([ChannelGate(video.hidden_dim, device=next(b.parameters()).device)
                              for b in video.blocks])
    ffn = nn.ModuleList([ChannelGate(video.ffn_dim, device=next(b.parameters()).device)
                         for b in video.blocks])
    handles = []
    for block, rg, fg in zip(video.blocks, residual, ffn):
        handles.append(block.register_forward_pre_hook(lambda module, args, gate=rg: (gate(args[0]), *args[1:])))
        handles.append(block.ffn[1].register_forward_hook(lambda module, args, output, gate=fg: gate(output)))
    return residual, ffn, handles


def masked_probe(prediction, mask, seed):
    generator = torch.Generator(device=prediction.device).manual_seed(seed)
    signs = torch.randint(0, 2, prediction.shape, device=prediction.device, generator=generator).float().mul_(2).sub_(1)
    valid = torch.ones_like(prediction, dtype=torch.float32)
    if mask is not None:
        valid = (~mask.bool()).unsqueeze(-1).expand_as(prediction).float()
    return (prediction.float() * signs * valid).sum() / valid.sum().clamp_min(1).sqrt()


def gradients(loss, gates, retain_graph=False):
    result = torch.autograd.grad(loss, [g.multiplier for g in gates], allow_unused=True, retain_graph=retain_graph)
    return [torch.zeros_like(g.multiplier) if value is None else value.float()
            for g, value in zip(gates, result)]


def task_scores(behaviour, demonstration):
    """Inputs [tasks, ..., channels]; scores normalized separately before combining."""
    if behaviour.shape != demonstration.shape or not torch.isfinite(behaviour).all() or not torch.isfinite(demonstration).all():
        raise ValueError('Invalid channel score arrays')
    def norm(x):
        if (x < 0).any():
            raise ValueError('Negative importance')
        return x / x.sum(-1, keepdim=True).clamp_min(1e-30)
    mixed = (norm(behaviour) + norm(demonstration)) * 0.5
    return mixed.mean(0) * 0.5 + mixed.max(0).values * 0.5


def select(score, count):
    if not torch.isfinite(score).all():
        raise ValueError('Nonfinite channel importance')
    if score.abs().sum() == 0:
        return evenly_spaced(len(score), count), True
    indices = torch.argsort(score, descending=True, stable=True)[:count].sort().values.tolist()
    return indices, False


def selection_from_statistics(behaviour, demonstration):
    # [task, layer, residual-or-FFN-channel], with common residual coordinates.
    residual = task_scores(behaviour['residual'].mean(1), demonstration['residual'].mean(1))
    ffns = task_scores(behaviour['ffn'], demonstration['ffn'])
    hidden, zero = select(residual, 2048)
    if zero:
        raise RuntimeError('No action sensitivity in the entire residual stream')
    selected, unused = [], []
    for index, score in enumerate(ffns):
        indices, empty = select(score, 8192)
        selected.append(indices)
        if empty:
            unused.append(index)
    return dict(schema='vpp2-width-selection-v1', method='action_sensitivity', hidden_indices=hidden,
                ffn_indices=selected, zero_sensitivity_ffn_layers=unused,
                behaviour_fraction=0.5, task_mean_fraction=0.5)
