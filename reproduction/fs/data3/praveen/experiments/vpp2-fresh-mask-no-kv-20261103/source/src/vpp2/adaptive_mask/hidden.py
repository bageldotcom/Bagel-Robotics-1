"""Reversible, isolated hidden-width trials within one mask-learning trajectory."""
import copy
from dataclasses import dataclass, field, asdict

from .controller import ranking
from .quality import overlap


@dataclass
class HiddenSearch:
    widths: list = field(default_factory=lambda: [5120, 4992, 4864])
    first_trial_update: int = 280
    last_trial_update: int = 1120
    trial_updates: int = 280
    rank_checks: int = 3
    rank_overlap: float = .995
    width: int = 5120
    trials: int = 0
    blocked: bool = False
    pending: dict | None = None
    candidates: list = field(default_factory=list)
    last_decision: str = 'collect_hidden_scores'

    def __post_init__(self):
        if (len(self.widths) < 2 or self.width != self.widths[0] and self.width not in self.widths
                or any(a <= b for a, b in zip(self.widths, self.widths[1:]))
                or self.trial_updates <= 0 or self.rank_checks < 2
                or not 0 < self.rank_overlap <= 1):
            raise ValueError('Invalid hidden search settings')

    @property
    def active(self):
        return self.pending is not None

    def state_dict(self):
        return copy.deepcopy(asdict(self))

    def observe(self, controller):
        next_index = self.widths.index(self.width) + 1
        if self.active or next_index == len(self.widths) or self.blocked:
            return
        count = self.widths[next_index]
        self.candidates.append(dict(width=count, indices=sorted(ranking(controller.hidden_scores)[:count])))
        self.candidates = self.candidates[-self.rank_checks:]

    def start(self, attempt, controller, inventory, budget, fitting, churn, quality_passed):
        """Call only after a complete accepted development check and parent snapshot."""
        if (self.active or self.blocked or not quality_passed or not fitting.get('passed')
                or not fitting.get('complete') or budget.stop_shrinking or budget.selection_hold
                or fitting.get('source')!='fitting_only'
                or not self.first_trial_update <= attempt <= self.last_trial_update
                or self.trials >= len(self.widths)-1 or len(self.candidates) < self.rank_checks
                or churn.get('hidden', 0.) > .005
                or churn['ffn_max'] > .05 or churn['cross_heads_max'] > .25):
            return False
        records = self.candidates
        if any(overlap(a['indices'], b['indices']) < self.rank_overlap
               for a, b in list(zip(records, records[1:])) + [(records[0], records[-1])]):
            return False
        anchor = controller.selection()
        if len(anchor['hidden_indices']) != self.width:
            raise ValueError('Hidden width/controller state mismatch')
        next_width = self.widths[self.widths.index(self.width)+1]
        if any(r['width'] != next_width for r in records):
            raise ValueError('Stale hidden candidate ranking history')
        self.pending = dict(start_attempt=attempt, previous_width=self.width,
                            anchor=anchor, cap=budget.cap,
                            previous_parameters=inventory.count(anchor))
        self.width = next_width
        self.trials += 1
        self.candidates.clear()
        self.last_decision = 'begin_isolated_hidden_trial'
        budget.authorized_updates = 0
        self.project(controller, inventory, budget.cap)
        return True

    def project(self, controller, inventory, cap):
        if not self.active:
            return controller.project(inventory, cap, self.width)
        # Only hidden identities may change in the trial. FFN/head selections,
        # not merely their widths, remain identical to the accepted parent.
        if cap != self.pending['cap']:
            raise ValueError('Capacity changed during isolated hidden trial')
        selection = copy.deepcopy(self.pending['anchor'])
        selection['hidden_indices'] = sorted(ranking(controller.hidden_scores)[:self.width])
        controller.set_selection(selection)
        count = inventory.cost(self.width,[len(x) for x in selection['ffn_indices']],
                               [len(x) for x in selection['cross_head_indices']])
        if count > cap or count >= self.pending['previous_parameters']:
            raise ValueError('Hidden trial did not reduce physical parameters')
        return dict(parameters=count, cap=cap, hidden=self.width,
                    ffn=[len(x) for x in selection['ffn_indices']],
                    cross_heads=[len(x) for x in selection['cross_head_indices']])

    def full_due(self, attempt):
        return self.active and attempt >= self.pending['start_attempt'] + self.trial_updates

    def assess(self, attempt, quality, fitting, full, churn):
        if not self.active:
            raise ValueError('No hidden trial to assess')
        if quality['resolved_violation'] or fitting.get('resolved_violation'):
            self.last_decision = 'reject_hidden_confirmed_harm'
        elif full and self.full_due(attempt):
            if quality['passed'] and fitting.get('passed') and churn.get('hidden', 0.) <= .005:
                self.last_decision = 'accept_hidden_width'
            else:
                self.last_decision = 'reject_hidden_unresolved_quality_or_turnover'
        else:
            self.last_decision = 'refine_isolated_hidden_trial'
        return self.last_decision

    def accept(self):
        if self.last_decision != 'accept_hidden_width':
            raise ValueError('Hidden trial has not passed its complete checks')
        self.pending = None
        self.candidates.clear()

    def restored(self, selection, reason='hidden_trial_rejected'):
        """The caller restores scores, optimizer, sampler, RNG and priority first."""
        self.width = len(selection['hidden_indices'])
        if self.width not in self.widths:
            raise ValueError('Restored width is outside the declared search')
        self.pending = None
        self.candidates.clear()
        self.blocked = True
        self.last_decision = reason
