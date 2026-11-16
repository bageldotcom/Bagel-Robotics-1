"""Fitting-only feedback; no additional policy forwards or reliability filtering."""
from collections import defaultdict
import math

from .quality import assess
from vpp2.full_mask.panel import seed_for


class FittingFeedback:
    """Episode/stratum means from matched pre-update predictions on fitting data.

    Store sufficient statistics rather than graphs or all visited windows. Global
    sample positions impose the same summation order after every distributed gather.
    Monitoring/audit rows are rejected, including on restart.
    """

    def __init__(self, fitting_rows, state=None):
        self.allowed = {r['id']: r for r in fitting_rows}
        if not self.allowed or any(r['role'] != 'selection_initial' for r in fitting_rows):
            raise ValueError('Feedback accepts fitting data only')
        self.groups = {}
        self.last_position = -1
        allowed_episodes = {(r['task'], r['suite'], r['episode']) for r in fitting_rows}
        if state is not None:
            self.last_position = state['last_position']
            for value in state['groups']:
                key = (value['task'], value['suite'], value['episode'], value['stratum'])
                if key[:3] not in allowed_episodes:
                    raise ValueError('Feedback resume contains non-fitting episodes')
                self.groups[key] = list(value['sums'])

    def add(self, records):
        for value in sorted(records, key=lambda r: r['position']):
            row = self.allowed.get(value['window_id'])
            if row is None or value['position'] <= self.last_position:
                raise ValueError('Non-fitting, repeated or reordered feedback sample')
            s = value['stratum']
            teacher, student = value['teacher_error'], value['student_error']
            if s not in range(4) or not all(math.isfinite(x) and x >= 0 for x in (teacher, student)):
                raise ValueError('Invalid fitting feedback')
            key = (row['task'], row['suite'], row['episode'], s)
            sums = self.groups.setdefault(key, [0., 0., 0])
            sums[0] += teacher
            sums[1] += student
            sums[2] += 1
            self.last_position = value['position']

    def state_dict(self):
        return dict(last_position=self.last_position, groups=[
            dict(task=t, suite=s, episode=e, stratum=n, sums=list(v))
            for (t, s, e, n), v in sorted(self.groups.items())])

    def report(self, quality_config, minimum_episodes=8):
        tasks = {r['task'] for r in self.allowed.values()}
        reports = {}
        complete = True
        settings = dict(quality_config)
        settings['alpha'] = settings.get('alpha', .05) / 4
        for stratum in range(4):
            rows = [dict(task=t, suite=s, episode=e, frame=-1, stratum=n,
                         teacher_error=a/count, student_error=b/count, samples=count)
                    for (t, s, e, n), (a, b, count) in sorted(self.groups.items()) if n == stratum]
            counts = defaultdict(int)
            for r in rows:
                counts[r['task']] += 1
            covered = set(counts) == tasks and min(counts.values(), default=0) >= minimum_episodes
            complete &= covered
            result = assess(rows, **settings) if rows else dict(passed=False, resolved_violation=False)
            reports[str(stratum)] = dict(quality=result, complete=covered, episodes_by_task=dict(counts))
        return dict(source='fitting_only', reliability_filtered=False,
                    complete=bool(complete), passed=bool(complete and all(x['quality']['passed'] for x in reports.values())),
                    resolved_violation=any(x['complete'] and x['quality']['resolved_violation'] for x in reports.values()),
                    strata=reports, windows=sum(v[2] for v in self.groups.values()))

    def clear(self):
        self.groups.clear()


def selection_churn(before, after):
    """Replacement fraction; planned differences in width alone are not churn."""
    def replacement(a, b):
        return 1 - len(set(a) & set(b)) / max(min(len(a), len(b)), 1)
    ffn = [replacement(a, b) for a, b in zip(before['ffn_indices'], after['ffn_indices'])]
    heads = [replacement(a, b) for a, b in zip(before['cross_head_indices'], after['cross_head_indices'])]
    return dict(ffn_max=max(ffn), cross_heads_max=max(heads), ffn_by_layer=ffn, cross_heads_by_layer=heads,
                hidden=replacement(before['hidden_indices'],after['hidden_indices']) if 'hidden_indices' in before else 0.)


def diagnostic_rows(panel, seed, check_index):
    """One fitting window per suite; rotate tasks so every task is represented."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in panel['windows']:
        if row['role'] == 'selection_initial':
            groups[row['suite']][row['task']].append(row)
    rows = []
    for suite, tasks in sorted(groups.items()):
        order = sorted(tasks, key=lambda t: seed_for(seed, 'fitting-gradient-task', suite, t))
        task = order[check_index % len(order)]
        choices = sorted(tasks[task], key=lambda r: seed_for(seed, 'fitting-gradient-window', r['id']))
        rows.append(choices[(check_index // len(order)) % len(choices)])
    return rows
