"""Action-step metrics with explicit execution timing and physical units."""
from collections import defaultdict
import numpy as np
from .contracts import critical_mask


def native_queries(length, replan=24):
    if type(length) is not int or length <= 0 or replan != 24:
        raise ValueError('Use positive episode length and unchanged native replan=24')
    return list(range(0, length, replan))


def executed_mask(start, length, horizon=32, replan=24):
    if not 0 <= start < length or horizon != 32 or replan != 24:
        raise ValueError('Native horizon/replanning contract changed')
    offsets = np.arange(horizon)
    return (offsets < replan) & (start + offsets < length)


def action_errors(pred_normalized, gt_normalized, pred_physical, gt_physical):
    arrays = [np.asarray(x, dtype=np.float64) for x in
              (pred_normalized, gt_normalized, pred_physical, gt_physical)]
    if any(x.shape != arrays[0].shape or x.ndim != 2 or x.shape[1] != 16 for x in arrays):
        raise ValueError('Expected aligned [horizon,16] action arrays')
    if any(not np.isfinite(x).all() for x in arrays):
        raise ValueError('Nonfinite prediction/target must remain an explicit evaluation failure')
    p, t, pp, tt = arrays
    errors = {'normalized_mse': np.square(p - t).mean(-1)}
    for name, off in [('left', 0), ('right', 8)]:
        errors[f'{name}_translation_mm'] = np.linalg.norm(pp[:, off:off+3] - tt[:, off:off+3], axis=-1) * 1000
        pq = pp[:, off+3:off+7]; tq = tt[:, off+3:off+7]
        pn = np.linalg.norm(pq, axis=-1); tn = np.linalg.norm(tq, axis=-1)
        if (tn < 1e-8).any():
            raise ValueError('Invalid ground-truth quaternion')
        dot = np.abs((pq * tq).sum(-1)) / np.maximum(pn * tn, 1e-16)
        angle = 2 * np.arccos(np.clip(dot, 0., 1.)) * 180 / np.pi
        angle[pn < 1e-8] = 180.
        errors[f'{name}_rotation_degrees'] = angle
        errors[f'{name}_invalid_rotation'] = (pn < 1e-8).astype(np.float64)
        errors[f'{name}_gripper_absolute_error'] = np.abs(pp[:, off+7] - tt[:, off+7])
    return errors


def summarize_values(errors, mask):
    mask = np.asarray(mask, dtype=bool); count = int(mask.sum())
    return dict(count=count, sums={k: float(v[mask].sum()) for k, v in errors.items()},
                maxima={k: float(v[mask].max()) if count else None for k, v in errors.items()})


def score_chunk(start, length, errors, annotation=None):
    horizon = len(next(iter(errors.values())))
    if horizon != 32:
        raise ValueError('Expected native 32-action prediction')
    executed = executed_mask(start, length)
    result = {'all_executed': summarize_values(errors, executed)}
    if annotation is None:
        return dict(groups=result, critical_labels_available=False)
    if annotation.get('status') != 'verified':
        raise ValueError('Draft labels cannot produce a critical validation score')
    intervals = annotation['intervals']
    critical = critical_mask(start, horizon, length, intervals) & executed
    result['critical_executed'] = summarize_values(errors, critical)
    result['ordinary_executed'] = summarize_values(errors, executed & ~critical)
    for kind in sorted({x['kind'] for x in intervals}):
        result['phase/' + kind] = summarize_values(errors, critical_mask(start, horizon, length, intervals, kind) & executed)
    for item in intervals:
        result['interval/' + item['id']] = summarize_values(errors, critical_mask(start, horizon, length, [item]) & executed)
    return dict(groups=result, critical_labels_available=True)


def aggregate(records):
    """Reduce action steps within episodes; average episodes, then tasks, equally.

    Intervals are episode-local. They are reported for worst-interval inspection,
    never pooled as if same-named intervals in different episodes were identical.
    """
    episode = {}; seen = set()
    for row in records:
        key = (row['variant'], row['episode'], row['frame'])
        if key in seen:
            raise ValueError('Duplicate evaluation query would inflate coverage')
        seen.add(key)
        for group, values in row['scores']['groups'].items():
            k = (row['variant'], row['task'], row['suite'], row['episode'], group)
            state = episode.setdefault(k, dict(count=0, sums=defaultdict(float), maxima={}))
            state['count'] += values['count']
            for metric, value in values['sums'].items():
                state['sums'][metric] += value
                maximum = values['maxima'][metric]
                if maximum is not None:
                    state['maxima'][metric] = max(state['maxima'].get(metric, -np.inf), maximum)
    per_episode = []
    for (variant, task, suite, ep, group), values in sorted(episode.items()):
        n = values['count']
        per_episode.append(dict(variant=variant, task=task, suite=suite, episode=ep, group=group,
                                steps=n, metrics={k: v/n for k, v in values['sums'].items()} if n else None,
                                maxima=values['maxima']))
    def reduce(rows, keys):
        grouped = defaultdict(list)
        for row in rows:
            if row['metrics'] is not None and not row['group'].startswith('interval/'):
                grouped[tuple(row[k] for k in keys)].append(row)
        return [{**dict(zip(keys, key)), 'members': len(rs),
                 'metrics': {m: float(np.mean([r['metrics'][m] for r in rs])) for m in rs[0]['metrics']}}
                for key, rs in sorted(grouped.items())]
    tasks = reduce(per_episode, ('variant', 'task', 'suite', 'group'))
    suites = reduce(tasks, ('variant', 'suite', 'group'))
    return dict(per_episode=per_episode, by_task=tasks, by_suite=suites,
                interpretation='Offline final-action errors on demonstrated observations; not rollout success.',
                timing='Episode queries at 0,24,...; only valid executed offsets 0..23 are scored.',
                dtw_enabled=False, temporal_ensemble_enabled=False)
