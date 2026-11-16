"""Defer detached logging transfers without changing optimizer arithmetic.

Keep the old FP32 scalar conversion and Python summation order. Vectorizing the
mean in FP32 would change some reported values, so only the transfer is batched.
"""
import torch


class DeferredMetrics:
    def __init__(self, device):
        self.device = device
        self.keys = None
        self.scalars = []
        self.actions = []
        self.labels = []

    def append(self, metrics, details, rows):
        keys = tuple(metrics)
        if self.keys is None:
            self.keys = keys
        elif keys != self.keys:
            raise ValueError('Metric columns changed within an optimizer update')
        values = torch.stack([torch.as_tensor(metrics[k], device=self.device).detach().float()
                              for k in keys])
        if values.ndim != 1:
            raise ValueError('Expected scalar logging metrics')
        actions = torch.stack([details[k].detach()
                               for k in ('student_error', 'teacher_error', 'reliability')], -1)
        if actions.shape != (len(rows), 3):
            raise ValueError('Per-window metrics and sample labels disagree')
        self.scalars.append(values)
        self.actions.append(actions)
        self.labels.extend((r['task'], r['suite']) for r in rows)

    def finish(self):
        if not self.scalars:
            raise ValueError('Cannot summarize an empty optimizer update')
        # Exactly two device-to-host transfers per update instead of two per
        # microbatch. Retained tensors are small and have no autograd graph.
        values = torch.stack(self.scalars).cpu().tolist()
        actions = torch.cat(self.actions, dim=0).cpu().tolist()
        row = {k: sum(v[i] for v in values)/len(values)
               for i, k in enumerate(self.keys)}
        tasks = [dict(task=task, suite=suite, student_error=v[0],
                      teacher_error=v[1], reliability=v[2])
                 for (task, suite), v in zip(self.labels, actions)]
        self.scalars.clear()
        self.actions.clear()
        self.labels.clear()
        self.keys = None
        return row, tasks
