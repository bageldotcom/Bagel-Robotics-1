"""Paired episode-level noninferiority; absence of significance is insufficient."""
from collections import defaultdict
import math
import numpy as np
from scipy.stats import t as t_distribution

def assess(rows,reference_key='teacher_error',student_key='student_error',task_margin=.05,suite_margin=.02,absolute_floor=1e-8,alpha=.05):
    grouped=defaultdict(list);seen=set()
    for r in rows:
        key=(r['task'],r['episode'],r['frame'],r.get('stratum',-1))
        if key in seen:raise ValueError('Duplicated quality observations')
        seen.add(key)
        a,b=r[reference_key],r[student_key]
        if not all(math.isfinite(x) and x>=0 for x in (a,b)):raise ValueError('Invalid quality errors')
        grouped[r['task'],r['suite'],r['episode']].append((a,b))
    tasks=defaultdict(list)
    for (task,suite,ep),values in grouped.items():tasks[task,suite].append(np.mean(values,axis=0))
    if not tasks:raise ValueError('Empty quality panel')
    task_reports={};suite_groups=defaultdict(list)
    ncomparisons=len(tasks)+len({s for t,s in tasks})
    def interval(mean,variance,df,ref,margin):
        if df<1:return dict(passed=False,upper=None,lower=None,allowance=max(margin*ref,absolute_floor),delta=mean)
        radius=float(t_distribution.ppf(1-alpha/ncomparisons,df))*math.sqrt(max(variance,0))
        allowance=max(margin*ref,absolute_floor)
        return dict(passed=mean+radius<=allowance,upper=mean+radius,lower=mean-radius,
                    allowance=allowance,delta=mean,clearly_passing=mean+radius<=.5*allowance,
                    resolved_violation=mean-radius>allowance)
    for (task,suite),values in sorted(tasks.items()):
        a=np.asarray(values);d=a[:,1]-a[:,0];n=len(a);ref=float(a[:,0].mean())
        var=float(np.var(d,ddof=1)/n) if n>1 else 0.
        report=interval(float(d.mean()),var,n-1,ref,task_margin)
        report.update(episodes=n,reference=ref,student=float(a[:,1].mean()),ratio=float(a[:,1].mean())/max(ref,absolute_floor),suite=suite)
        task_reports[task]=report;suite_groups[suite].append((report,var,n-1))
    suites={}
    for suite,items in suite_groups.items():
        n=len(items);ref=float(np.mean([r['reference'] for r,v,d in items]));mean=float(np.mean([r['delta'] for r,v,d in items]))
        var=sum(v for r,v,d in items)/n**2;df=min(d for r,v,d in items)
        suites[suite]=interval(mean,var,df,ref,suite_margin)
        suites[suite].update(reference=ref,student=float(np.mean([r['student'] for r,v,d in items])))
    reports=list(task_reports.values())+list(suites.values())
    return dict(passed=all(r['passed'] for r in reports),clearly_passing=all(r.get('clearly_passing',False) for r in reports),
        resolved_violation=any(r.get('resolved_violation',False) for r in reports),by_task=task_reports,by_suite=suites,
        paired_episode_level=True,multiplicity='Bonferroni tasks + suites',policy_retention_established=False)

def overlap(a,b):
    if len(a)!=len(b):return 0.
    return len(set(a)&set(b))/max(len(a),1)

def stable(records):
    """Five checks span 280 updates. Compare adjacent and first-to-last identities."""
    if len(records)<5:return False
    rs=records[-5:]
    # A protocol change starts a new stability-evidence interval without rewriting history.
    if len({r.get('quality_gate_mode','action_and_video_v1') for r in rs})!=1:return False
    if rs[-1]['lineage_update']-rs[0]['lineage_update']<280:return False
    if len({r['cap'] for r in rs})!=1:return False
    if not all(r.get('screen_clear',r.get('quality_passed',False)) and r['gradients_active'] for r in rs):return False
    for before,after in list(zip(rs,rs[1:]))+[(rs[0],rs[-1])]:
        a,b=before['selection'],after['selection']
        if overlap(a['hidden_indices'],b['hidden_indices'])<.995:return False
        if any(overlap(x,y)<.99 for x,y in zip(a['ffn_indices'],b['ffn_indices'])):return False
        if a['cross_head_indices']!=b['cross_head_indices']:return False
    return True
