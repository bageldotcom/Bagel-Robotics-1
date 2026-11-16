"""Read-only matched action-metric comparison after the no-K/V pilot."""
import json
from pathlib import Path
from statistics import mean

ROOT=Path('/data3/praveen/experiments/vpp2-recovery55-no-kv-20261102')
HIST=Path('/data3/praveen/experiments/vpp2-selected-mask-recovery-curated30k-20261101')

def read(p):return json.loads(p.read_text())
def write(p,d):
    q=p.with_suffix('.partial');q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)
def compare(a,b):
    return dict(student_gt_action_mse=a['student_error'],reference_student_gt_action_mse=b['student_error'],
        absolute_change=a['student_error']-b['student_error'],
        relative_change_percent=100*(a['student_error']/b['student_error']-1) if b['student_error']>0 else None,
        teacher_gt_action_mse=a['teacher_error'])
def windows(d):return sorted((r['window_id'],r['episode'],r['frame'],r['executed_steps']) for r in d['rows'])
def macro(d):
    t=[v for k,v in d['actions'].items() if k.startswith('task/')]
    return {k:mean(x[k] for x in t) for k in ('student_error','teacher_error')}

def main():
    done=read(ROOT/'control/review_ready.json')
    if done['update']!=250 or not done['checkpoint_saved']:raise ValueError('Pilot not complete')
    a=read(ROOT/'recovery/diagnostics/native_000250.json')
    start=read(ROOT/'recovery/diagnostics/native_000055.json')
    old=read(HIST/'recovery/diagnostics/native_000250.json')
    if not windows(a)==windows(start)==windows(old):raise ValueError('Unmatched validation windows')
    result=dict(scope='Fixed 140-window offline native-action monitoring panel; no simulation results',
        two_changes='No K/V matching and learning rate1e-6; cannot isolate a causal effect of K/V',
        historical250_weights_available=False,checkpoint250=str(ROOT/'recovery/checkpoints/update_000250'),
        new_optimizer_updates=195,best=read(ROOT/'recovery/best_models.json')['best'],
        overall_vs55=compare(macro(a),macro(start)),overall_vs_historical250=compare(macro(a),macro(old)),
        per_group_vs55={k:compare(v,start['actions'][k]) for k,v in a['actions'].items()},
        per_group_vs_historical250={k:compare(v,old['actions'][k]) for k,v in a['actions'].items()},
        student_teacher_action_mse=dict(at55=start['action_agreement'],at250=a['action_agreement'],
            historical250=None,historical250_missing_reason='Direct prediction disagreement was not recorded in the old run'),
        fallback_launched=False,original_replay_resume_update=94)
    task_changes=[v['absolute_change'] for k,v in result['per_group_vs55'].items() if k.startswith('task/')]
    result['tasks_vs55']=dict(lower_gt_error=sum(v<0 for v in task_changes),higher_gt_error=sum(v>0 for v in task_changes),unchanged=sum(v==0 for v in task_changes))
    write(ROOT/'control/review_250.json',result)
    print(json.dumps({k:result[k] for k in ('overall_vs55','overall_vs_historical250','tasks_vs55')}))

if __name__=='__main__':main()
