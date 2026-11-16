"""Single trajectory: fast score learning, slow fitting-driven capacity decisions."""
from dataclasses import dataclass, asdict, field


@dataclass
class Budget:
    original: int
    target: int = 10_000_000_000
    information_updates: int = 70
    milestones: list = field(default_factory=lambda: [[560, 14_000_000_000], [1120, 12_000_000_000], [1680, 10_000_000_000]])
    stability_updates: int = 280
    extra_updates: int = 280
    early_full_update: int = 280
    feedback_every: int = 70
    progress: int = 0
    attempt: int = 0
    cap: int = 0
    phase: str = 'collect'
    extra_used: bool = False
    rescue_used: bool = False
    rescue_remaining: int = 0
    hold_age: int = 0
    accepted_cap: int = 0
    authorized_updates: int = 0
    selection_hold: bool = False
    stop_shrinking: bool = False
    last_feedback: str = 'collect'

    def __post_init__(self):
        if not self.cap: self.cap = self.original
        if not self.accepted_cap: self.accepted_cap = self.original
        if self.milestones[-1][1] != self.target: raise ValueError('Target/milestone mismatch')
        knots = [[self.information_updates, self.original]] + self.milestones
        if any(b[0] <= a[0] or b[1] >= a[1] for a, b in zip(knots, knots[1:])):
            raise ValueError('Invalid progressive schedule')
        if self.information_updates % self.feedback_every or self.stability_updates % self.feedback_every:
            raise ValueError('Controller and monitoring cadence mismatch')

    def state_dict(self): return asdict(self)

    @property
    def planned_updates(self): return self.milestones[-1][0] + self.stability_updates

    @property
    def maximum_updates(self): return self.planned_updates + self.extra_updates

    def scheduled_cap(self):
        if self.progress <= self.information_updates: return self.original
        knots = [[self.information_updates, self.original]] + self.milestones
        for a, b in zip(knots, knots[1:]):
            if self.progress <= b[0]:
                return a[1] - (a[1] - b[1]) * (self.progress - a[0]) // (b[0] - a[0])
        return self.target

    def advance(self, hold_capacity=False):
        """The size clock advances only within a fitting-approved segment."""
        old_cap = self.cap
        self.attempt += 1
        self.rescue_remaining = max(0, self.rescue_remaining - 1)
        if self.progress < self.information_updates:
            self.progress += 1
            self.phase = 'collect'
        elif (self.authorized_updates > 0 and not hold_capacity and not self.stop_shrinking and not self.selection_hold
              and self.attempt <= self.milestones[-1][0] and self.rescue_remaining == 0):
            self.progress += 1
            self.authorized_updates -= 1
            self.cap = self.scheduled_cap()
            self.phase = 'reduce'
        else:
            self.phase = 'settle' if self.attempt >= self.milestones[-1][0] or self.stop_shrinking else 'hold'
        self.hold_age = self.hold_age + 1 if self.cap == old_cap else 0

    def observe_fitting(self, report, churn, max_ffn_churn=.05, max_head_churn=.25, max_hidden_churn=.005, hold_capacity=False):
        """Authorize at most one segment; uncertainty never authorizes shrinking."""
        if self.attempt % self.feedback_every:
            raise ValueError('Capacity decisions must use a complete feedback interval')
        self.authorized_updates = 0
        if self.attempt < self.information_updates: decision = 'collect'
        elif self.stop_shrinking or self.attempt >= self.milestones[-1][0] or self.cap <= self.target:
            decision = 'hold_final_capacity'
        elif hold_capacity: decision = 'hold_isolated_hidden_trial'
        elif self.selection_hold or self.rescue_remaining: decision = 'hold_selection_check'
        elif report.get('source') != 'fitting_only' or not report.get('complete'):
            decision = 'hold_insufficient_fitting_coverage'
        elif report.get('resolved_violation'): decision = 'hold_fitting_harm'
        elif not report.get('passed'): decision = 'hold_fitting_uncertainty'
        elif churn['ffn_max'] > max_ffn_churn or churn['cross_heads_max'] > max_head_churn or churn.get('hidden',0.) > max_hidden_churn:
            decision = 'hold_channel_turnover'
        else:
            self.authorized_updates = self.feedback_every
            decision = 'authorize_gradual_reduction'
        self.last_feedback = decision
        return decision

    def full_due(self):
        # Fixed cost: baseline plus at most six checks; fitting holds add no passes.
        return self.attempt in {0, self.early_full_update, *[x[0] for x in self.milestones],
                                self.planned_updates, self.maximum_updates}

    def decide(self, passed, stable=False, coverage=False, resolved_violation=False, extra_quality_check=False):
        """A development selection check; uncertainty and confirmed harm differ."""
        if (not self.full_due() and not extra_quality_check) or self.attempt == 0:
            raise ValueError('Selection decision outside a scheduled full check')
        if passed and resolved_violation: raise ValueError('Contradictory quality evidence')
        if passed:
            self.accepted_cap = self.cap
            self.selection_hold = False
            self.rescue_remaining = 0
        else:
            self.authorized_updates = 0
            self.selection_hold = True
            if self.attempt >= self.maximum_updates:
                return 'stop_quality_failed' if resolved_violation else 'stop_quality_unresolved'
            if resolved_violation:
                if not self.rescue_used:
                    self.rescue_used = True
                    self.rescue_remaining = self.extra_updates
                    return 'rescue_confirmed_harm_once'
                if self.rescue_remaining == 0:
                    self.cap = self.accepted_cap
                    self.stop_shrinking = True
                    self.hold_age = 0
                    return 'restore_parent_and_stop_shrinking'
        if self.attempt >= self.planned_updates:
            if passed and stable and coverage: return 'finish'
            if self.attempt < self.maximum_updates:
                self.extra_used = True
                self.stop_shrinking = True
                return 'extend_final_check_once'
            return 'stop_not_stable' if passed else 'stop_quality_failed' if resolved_violation else 'stop_quality_unresolved'
        return 'accept' if passed else 'hold_confirmed_harm' if resolved_violation else 'hold_quality_uncertain'
