"""Continuous masks; one canonical binary selection for accounting and export."""
import math
import torch
from torch import nn


def schedule(update):
    if update < 0:
        raise ValueError('Negative update')
    sharpen = min(max((update - 500) / 500, 0.), 1.)
    return dict(alpha=min(update / 200, 1.), temperature=1 + 9 * sharpen,
                entropy=.01 * sharpen, discreteness=.01 * sharpen,
                teacher=min(update / 100, 1.))


def ranked_indices(values, count):
    """Stable ties prefer the lower original channel index."""
    if not torch.isfinite(values).all():
        raise FloatingPointError('Nonfinite mask score')
    return sorted(torch.argsort(values.detach().cpu(), descending=True, stable=True)[:count].tolist())


def binary_indices(values, floor, pack):
    count = int((values.detach() >= 0).sum())
    required = min(values.numel(), math.ceil(max(count, floor) / pack) * pack)
    return ranked_indices(values, required), required - count


class Masks(nn.Module):
    def __init__(self, geometry, rankings, preferred):
        super().__init__()
        self.geometry = geometry
        shapes = dict(hidden_scores=(geometry.hidden,),
                      ffn_scores=(geometry.layers-1, geometry.ffn),
                      cross_scores=(geometry.layers-1, geometry.heads))
        for key, shape in shapes.items():
            source = rankings[key].detach().cpu().float()
            if tuple(source.shape) != shape:
                raise ValueError('Wrong warm-start mask geometry: ' + key)
            target = torch.full(shape, -1.5, dtype=torch.float32)
            count = preferred[key]
            if not 0 < count <= shape[-1]:
                raise ValueError('Invalid initial preferred count')
            if source.ndim == 1:
                target[ranked_indices(source, count)] = 1.5
            else:
                for row in range(len(source)):
                    target[row, ranked_indices(source[row], count)] = 1.5
            setattr(self, key, nn.Parameter(target))

    def families(self):
        return (self.hidden_scores, self.ffn_scores, self.cross_scores)

    def probabilities(self, update):
        temperature = schedule(update)['temperature']
        return tuple(torch.sigmoid(temperature * p) for p in self.families())

    def selection(self):
        g = self.geometry
        selection = g.full()
        selection['hidden_indices'], hidden_added = binary_indices(self.hidden_scores, g.hidden_floor, g.pack)
        ffn_added, cross_added = [], []
        for i in range(g.layers-1):
            selection['ffn_indices'][i], added = binary_indices(self.ffn_scores[i], g.ffn_floor, g.pack)
            ffn_added.append(added)
            selection['cross_head_indices'][i], added = binary_indices(self.cross_scores[i], g.head_floor, 1)
            cross_added.append(added)
        g.validate(selection)
        return selection, dict(hidden=hidden_added, ffn=ffn_added, cross_heads=cross_added)

    def gates(self, update, mode='soft'):
        if mode == 'open':
            return tuple(torch.ones_like(p) for p in self.families())
        if mode == 'soft':
            alpha = schedule(update)['alpha']
            return tuple((1-alpha) + alpha*m for m in self.probabilities(update))
        if mode != 'binary':
            raise ValueError(mode)
        s, _ = self.selection()
        result = [torch.zeros_like(p) for p in self.families()]
        result[0][s['hidden_indices']] = 1
        for i in range(self.geometry.layers-1):
            result[1][i, s['ffn_indices'][i]] = 1
            result[2][i, s['cross_head_indices'][i]] = 1
        return tuple(result)

    def regularizers(self, update):
        # Each family contributes equally; layers within FFN/head families do too.
        probabilities = self.probabilities(update)
        entropy = torch.stack([(-(p.clamp(1e-7,1-1e-7)*p.clamp(1e-7,1-1e-7).log()
            + (1-p).clamp(1e-7,1-1e-7)*(1-p).clamp(1e-7,1-1e-7).log())).mean()/math.log(2)
            for p in probabilities]).mean()
        discreteness = torch.stack([(2*torch.minimum(p, 1-p)).mean() for p in probabilities]).mean()
        return entropy, discreteness


class SoftCost:
    """Continuous extension of exact tensor slicing, including hidden-width products.

    At full FFN/head widths every native tensor has at most two hidden axes.
    Fit that integer quadratic using the SAME meta-tensor slicer as export, then
    subtract the exact FFN and cross-head contributions. No empirical cost fit.
    """
    def __init__(self, inventory):
        self.inventory = inventory
        g = inventory.geometry
        # Count meta shapes directly, allowing arbitrary H only for the algebra.
        from bagel_robotics._internal.adaptive_mask.geometry import slice_parameter
        def count(h):
            s = g.full(); s['hidden_indices'] = list(range(h))
            return sum(slice_parameter(k, torch.empty(v, device='meta'), s, g)[0].numel()
                       for k, v in inventory.shapes.items())
        f1, f2, f3 = (count(i) for i in (1, 2, 3))
        if (f3 - 2*f2 + f1) % 2:
            raise ValueError('Unexpected non-integer hidden cost polynomial')
        self.a = (f3-2*f2+f1)//2
        self.b = f2-f1-3*self.a
        self.c = f1-self.a-self.b
        if self.a*g.hidden*g.hidden + self.b*g.hidden + self.c != inventory.original:
            raise ValueError('Soft-cost polynomial does not reproduce full inventory')

    def parameters(self, gates):
        h, f, c = (x.double() for x in gates)
        g = self.inventory.geometry
        width = h.sum()
        # Terminal FFN/conditioning heads are never masked.
        return (self.a*width.square() + self.b*width + self.c
                - (g.ffn-f.sum(-1)).sum()*(2*width+1)
                - (g.heads-c.sum(-1)).sum()*g.head_dim*(6*width+8))

    def loss(self, gates):
        return (self.parameters(gates)/self.inventory.original).float()
