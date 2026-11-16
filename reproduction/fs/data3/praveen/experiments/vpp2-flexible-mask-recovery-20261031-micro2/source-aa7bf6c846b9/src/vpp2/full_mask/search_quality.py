"""Action-quality selection and conservative convergence, independent of GPU execution."""
from collections import defaultdict
import math
import numpy as np
from scipy.stats import t as student_t


def schedule_end(config):
    m=config['mask'];t=config['train']
    return max(m['entry_updates'],2*m['budget_ramp_updates'],t['kv_ramp_updates'],t['minimum_stop_update'])


def stable_history(history, update, config):
    end=schedule_end(config);count=config['train']['stable_checks']
    # Both endpoints must be after all mask schedules finish.
    eligible=[x for x in history if x.get('previous_update',-1)>=end]
    recent=eligible[-count:]
    return update>=end and len(recent)==count and all(x['stable'] for x in recent)


def quality_summary(rows, *, alpha=.05):
    """Equal task/episode weighting. CI is an offline diagnostic, not rollout proof.

    Per-task one-sided t intervals use independent episode means, Bonferroni
    adjusted over tasks. No arbitrary percentage of action degradation is hidden.
    """
    if not rows:raise ValueError('Empty quality panel')
    grouped=defaultdict(list);seen=set()
    for r in rows:
        key=(r['task'],r['episode'],r['frame'])
        if key in seen:raise ValueError('Duplicate action-quality window')
        seen.add(key)
        if not all(math.isfinite(r[k]) and r[k]>=0 for k in ('teacher_error','hard_error')):
            raise ValueError('Invalid action-quality error')
        grouped[(r['task'],r['suite'],r['episode'])].append(r)
    task_groups=defaultdict(list)
    for (task,suite,episode),rs in grouped.items():
        task_groups[(task,suite)].append((episode,float(np.mean([r['teacher_error'] for r in rs])),float(np.mean([r['hard_error'] for r in rs]))))
    tasks={};suites=defaultdict(list)
    for (task,suite),values in sorted(task_groups.items()):
        a=np.array([[x[1],x[2]] for x in values]);delta=a[:,1]-a[:,0];n=len(a)
        reference=float(a[:,0].mean());hard=float(a[:,1].mean());mean=float(delta.mean())
        se=float(delta.std(ddof=1)/math.sqrt(n)) if n>1 else None
        quantile=float(student_t.ppf(1-alpha/len(task_groups),n-1)) if n>1 else None
        lower=mean-quantile*se if n>1 else None
        upper=mean+quantile*se if n>1 else None
        ratio=hard/max(reference,1e-8)
        entry=dict(suite=suite,episodes=n,teacher_error=reference,hard_error=hard,
            ratio_to_teacher=ratio,paired_delta=mean,paired_standard_error=se,
            adjusted_lower=lower,adjusted_upper=upper,
            resolved_regression=lower is not None and lower>0,
            noninferiority_supported=upper is not None and upper<=0)
        tasks[task]=entry;suites[suite].append(entry)
    return dict(by_task=tasks,by_suite={s:dict(mean_task_ratio=float(np.mean([r['ratio_to_teacher'] for r in rs])),
        worst_task_ratio=max(r['ratio_to_teacher'] for r in rs)) for s,rs in suites.items()},
        mean_task_ratio=float(np.mean([x['ratio_to_teacher'] for x in tasks.values()])),
        worst_task_ratio=max(x['ratio_to_teacher'] for x in tasks.values()),
        resolved_regressions=[t for t,x in tasks.items() if x['resolved_regression']],
        no_resolved_regressions=all(x['adjusted_lower'] is not None and not x['resolved_regression'] for x in tasks.values()),
        noninferiority_supported=all(x['noninferiority_supported'] for x in tasks.values()),
        interpretation='Paired episode-level action errors; one-sided t intervals adjusted across tasks. Absence of a resolved regression is not equivalence or simulation success.',
        alpha=alpha,selection_uses_monitoring_data=True,policy_retention_established=False)


def better(candidate, incumbent):
    if incumbent is None:return True
    # Primary: equal-task normalized action MSE. Worst task breaks exact ties.
    return (candidate['mean_task_ratio'],candidate['worst_task_ratio']) < (incumbent['mean_task_ratio'],incumbent['worst_task_ratio'])


def sensitivity_check(previous,current,config,overlap):
    """Do not demand agreement in numerical dust; do not silently accept noisy signal."""
    cfg=config['train'];default=cfg['suite_sensitivity_top_overlap_min']
    if not cfg.get('noise_aware_sensitivity',False):
        h=overlap(previous['hidden_top'],current['hidden_top'])
        f=[overlap(a,b) for a,b in zip(previous['ffn_top'],current['ffn_top'])]
        return dict(passed=h>=default and min(f)>=default,hidden_overlap=h,ffn_overlap=f)
    now=current['sensitivity_groups'];old=previous['sensitivity_groups']
    if len(now)!=len(old):raise ValueError('Changed sensitivity groups')
    checks=[];minimum=cfg['sensitivity_half_overlap_min']
    for a,b in zip(old,now):
        if a['name']!=b['name']:raise ValueError('Changed sensitivity order')
        temporal=overlap(a['top'],b['top'])
        inactive=not a['active'] and not b['active']
        reliable=a['half_overlap']>=minimum and b['half_overlap']>=minimum
        threshold=max(minimum,min(default,a['half_overlap'],b['half_overlap']))
        passed=inactive or (a['active'] and b['active'] and reliable and temporal>=threshold)
        checks.append(dict(group=b['name'],active=b['active'],negligible_in_both=inactive,
            temporal_overlap=temporal,required_overlap=threshold,half_overlap=b['half_overlap'],passed=bool(passed)))
    # An entirely absent action gradient is a failure, not perfect stability.
    return dict(passed=any(x['active'] for x in checks) and all(x['passed'] for x in checks),groups=checks)
