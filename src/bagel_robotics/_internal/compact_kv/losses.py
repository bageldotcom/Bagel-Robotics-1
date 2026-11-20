"""Per-example losses; accumulation/rank partitioning cannot change weighting."""
import torch
from torch.utils.checkpoint import checkpoint


def action_error(prediction, target, is_pad=None):
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Action predictions and targets must have identical [B,T,D] shape")
    error = (prediction.float() - target.detach().float()).square().mean(-1)
    if is_pad is None:
        return error.mean(-1)
    if is_pad.shape != error.shape:
        raise ValueError("Action mask must match [B,T]")
    valid = (~is_pad.bool()).float()
    return (error * valid).sum(-1) / valid.sum(-1).clamp_min(1)


def kv_terms(student, teacher, num_heads=40, floor=1e-6):
    if len(student) != len(teacher) or not student:
        raise ValueError("Teacher and student must expose the same nonempty cache slots")
    layers = []
    for left, right in zip(student, teacher):
        terms = []
        for key in ("k", "v"):
            a, b = left[key], right[key]
            if a.shape != b.shape or a.ndim != 3 or a.shape[-1] % num_heads:
                raise ValueError("Mismatched K/V geometry")
            # Preserve raw magnitude. Normalize using this example's teacher
            # energy; do not compute denominators across the microbatch/ranks.
            def term(a, b):
                a = a.float().reshape(*a.shape[:2], num_heads, -1)
                b = b.detach().float().reshape(*b.shape[:2], num_heads, -1)
                energy = b.square().mean((1, 3)).clamp_min(floor)
                return ((a - b).square().mean((1, 3)) / energy).mean(-1)
            # Recompute cheap reductions in backward instead of retaining 40
            # layers of FP32 difference tensors alongside both BF16 caches.
            terms.append(checkpoint(term, a, b, use_reentrant=False)
                         if torch.is_grad_enabled() and a.requires_grad else term(a, b))
        layers.append(torch.stack(terms, dim=-1))
    return torch.stack(layers, dim=1)


def kv_error(student, teacher, num_heads=40, floor=1e-6):
    by_layer = kv_terms(student, teacher, num_heads, floor).mean(-1)
    return by_layer.mean(1), by_layer


def kv_plane_terms(student, teacher, planes=5, num_heads=40, floor=1e-6):
    """Diagnostics only: [B, plane, layer, K/V], separate per-plane energy."""
    if len(student) != len(teacher) or not student:
        raise ValueError('Different cache inventories')
    values = []
    for left,right in zip(student,teacher):
        parts=[]
        for key in ('k','v'):
            a,b=left[key],right[key]
            if a.shape != b.shape or a.shape[1] % planes or a.shape[2] % num_heads:
                raise ValueError('Invalid temporal cache geometry')
            a=a.detach().float().reshape(a.shape[0],planes,-1,num_heads,a.shape[2]//num_heads)
            b=b.detach().float().reshape_as(a)
            energy=b.square().mean((2,4)).clamp_min(floor)
            parts.append(((a-b).square().mean((2,4))/energy).mean(-1))
        values.append(torch.stack(parts,-1))
    return torch.stack(values,2)
