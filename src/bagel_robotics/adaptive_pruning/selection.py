"""Stability decides readiness; teacher closeness never blocks recovery."""
import math
from collections import defaultdict


def jaccard(left, right):
    a, b = set(left), set(right)
    return len(a & b)/len(a | b) if a | b else 1.


def overlaps(a, b, geometry):
    values = []
    for key, total, kept_threshold, removed_threshold in (
        ('hidden_indices', geometry.hidden, .995, .95),
        ('ffn_indices', geometry.ffn, .99, .95),
        ('cross_head_indices', geometry.heads, 1., 1.)):
        left = [a[key]] if key == 'hidden_indices' else a[key][:-1]
        right = [b[key]] if key == 'hidden_indices' else b[key][:-1]
        for i, (x, y) in enumerate(zip(left, right, strict=True)):
            kept = jaccard(x,y)
            removed = jaccard(set(range(total))-set(x), set(range(total))-set(y))
            values.append(dict(family=key,layer=i,kept=kept,removed=removed,
                               passed=kept>=kept_threshold and removed>=removed_threshold))
    return values


def aggregate(rows):
    if not rows or len({r['window_id'] for r in rows}) != len(rows):
        raise ValueError('Missing/duplicate diagnostic windows')
    tasks, suites = defaultdict(list), defaultdict(list)
    for r in rows:
        if not all(math.isfinite(r[k]) and r[k] >= 0 for k in ('student_error','teacher_error')):
            raise FloatingPointError('Nonfinite action diagnostic')
        tasks[r['task']].append(r)
    result = {}
    metrics=('student_error','teacher_error')
    if all('student_teacher_error' in r for r in rows):metrics+=('student_teacher_error',)
    for task, values in tasks.items():
        summary = {k:sum(r[k] for r in values)/len(values) for k in metrics}
        result['task/'+task] = summary
        suite_names = {r['suite'] for r in values}
        if len(suite_names)!=1:
            raise ValueError('Task belongs to multiple suites')
        suites[next(iter(suite_names))].append(summary)
    for suite, values in suites.items():
        result['suite/'+suite] = {k:sum(r[k] for r in values)/len(values) for k in metrics}
    return result


def readiness(history, geometry, update, coverage_min):
    reasons = []
    last = history[-5:]
    if update < 2000: reasons.append('minimum_2000_updates')
    if coverage_min < 2: reasons.append('not_every_fitting_window_seen_twice')
    if len(last)<5 or last[-1]['update']-last[0]['update']!=1000 or last[-1]['update']!=update:
        reasons.append('need_five_checks_spanning_1000_updates')
    else:
        pairs = list(zip(last[:-1],last[1:])) + [(last[0],last[-1])]
        if any(not v['passed'] for a,b in pairs for v in overlaps(a['selection'],b['selection'],geometry)):
            reasons.append('channel_identities_still_changing')
        counts = [x['parameters'] for x in last]
        if (max(counts)-min(counts))/counts[-1] > .005:
            reasons.append('physical_size_still_changing')
        if len({tuple(x['regularizer_state']) for x in last})!=1:
            reasons.append('temperature_or_regularizer_still_changing')
        if any(not x['gradient_active'] for x in last):
            reasons.append('inactive_or_numerically_saturated_mask_learning')
        keys = set(last[0]['actions'])
        if any(set(x['actions'])!=keys for x in last):
            raise ValueError('Diagnostic task/suite coverage changed')
        deteriorating=[]
        for key in sorted(keys):
            baseline=(last[0]['actions'][key]['student_error']+last[1]['actions'][key]['student_error'])/2
            threshold=baseline+.05*max(baseline,1e-8)
            if all(x['actions'][key]['student_error']>threshold for x in last[-2:]):
                deteriorating.append(key)
        if deteriorating: reasons.append('persistent_action_deterioration:'+','.join(deteriorating))
    status = 'ready_for_recovery' if not reasons else 'selection_not_converged'
    return dict(status=status,update=update,reasons=reasons,ceiling_reached=update>=3000)


def candidate_key(candidate):
    actions = candidate['full_native_actions']
    suite = [v for k,v in actions.items() if k.startswith('suite/')]
    tasks = [v for k,v in actions.items() if k.startswith('task/')]
    if not suite or not tasks:
        raise ValueError('Incomplete finalist diagnostic')
    worst=max((v['student_error']-v['teacher_error'])/max(v['teacher_error'],1e-8) for v in suite)
    return worst,sum(v['student_error'] for v in tasks)/len(tasks),candidate['parameters'],-candidate['update']
