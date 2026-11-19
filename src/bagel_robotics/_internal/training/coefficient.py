"""Measure on complete GT gradients, cap once, then ramp; no default fallback."""
from dataclasses import dataclass, field, asdict
import math
import statistics


@dataclass
class Coefficient:
    cap: float = 1.0
    target_fraction: float = .10
    samples: dict = field(default_factory=lambda: {k: [] for k in ('hidden', 'ffn', 'cross')})
    measured: float | None = None
    applied: float | None = None
    resolved_update: int | None = None
    last_observed: int = 0
    unresolved: bool = False

    def __post_init__(self):
        if not math.isfinite(self.cap) or self.cap <= 0:
            raise ValueError('Coefficient cap must be finite and positive')
        if self.target_fraction != .10:
            raise ValueError('Gradient calibration fraction changed')

    def collecting(self, update):
        return self.applied is None and 193 <= update <= 232 and not self.unresolved

    def observe(self, update, gt_norms, size_norms):
        if not self.collecting(update) or update <= self.last_observed:
            raise ValueError('Invalid/repeated calibration update')
        self.last_observed = update
        for name, gt, size in zip(self.samples, gt_norms, size_norms, strict=True):
            if len(self.samples[name]) < 8 and all(math.isfinite(x) and x > 1e-12 for x in (gt, size)):
                value = self.target_fraction*gt/size
                if math.isfinite(value):
                    self.samples[name].append(dict(update=update, candidate=value, gt_norm=gt, size_norm=size))
        if update >= 200 and all(len(v) == 8 for v in self.samples.values()):
            self.measured = min(statistics.median(x['candidate'] for x in v) for v in self.samples.values())
            self.applied = min(self.measured, self.cap)
            self.resolved_update = update
        elif update == 232:
            self.unresolved = True

    def weight(self, update):
        if self.applied is None:
            return 0.
        return self.applied * min(max((update-self.resolved_update)/100, 0.), 1.)

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state(cls, state, cap):
        result = cls(**state)
        if result.cap != cap:
            raise ValueError('Cannot change coefficient cap on resume')
        if result.applied is not None and not (0 <= result.applied <= cap and math.isfinite(result.applied)):
            raise ValueError('Invalid saved coefficient')
        return result
