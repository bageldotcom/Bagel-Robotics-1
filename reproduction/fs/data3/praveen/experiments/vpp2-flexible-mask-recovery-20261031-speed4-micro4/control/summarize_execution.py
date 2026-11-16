import json,statistics,datetime
from pathlib import Path
root=Path('/data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-speed4-micro4')
prior=Path('/data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-speed3-micro4')
rows=[json.loads(line) for line in (root/'mask/metrics.jsonl').read_text().splitlines()]
actual={r['update']:r for r in rows};assert len(actual)==len(rows),'Duplicate update records'
latest=max(actual);assert latest>=52,'Wait for two previously unbenchmarked production updates'
expected=json.loads((prior/'speed_review/micro4/metrics.json').read_text())
checks=[]
for row in expected:
    for key in ('gt_action','teacher_video_gated','late_kv_gated','total','video_gradient_norm'):
        x,y=row[key],actual[row['update']][key]
        checks.append(dict(update=row['update'],metric=key,benchmark=x,production=y,relative_difference=abs(x-y)/max(abs(x),1e-8)))
max_relative=max(x['relative_difference'] for x in checks)
assert max_relative<=.02,('Unexpected production/benchmark deviation',max_relative)
speed=json.loads((prior/'checks/speed_micro4.json').read_text());tasks=json.loads((prior/'checks/speed_micro4_tasks.json').read_text());cal=json.loads((root/'checks/calibration_equivalence.json').read_text());gpu=json.loads((root/'checks/gpu.json').read_text())
assert speed['passed'] and tasks['passed'] and cal['passed'] and gpu['passed']
fresh=[r['seconds_per_update'] for r in rows if r['update']>=51][-10:]
seconds=statistics.mean(fresh)
report=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),identity=gpu['identity'],production_job='50039',continuations=['50040','50041'],latest_update=latest,
    resumed_checkpoint_update=47,benchmark_updates_discarded=True,global_batch=256,world_size=8,micro_batch=4,accumulation=8,calibration_subbatch=2,
    matched_baseline_seconds=speed['baseline_seconds'],matched_candidate_seconds=speed['candidate_seconds'],matched_throughput_speedup=speed['speedup'],
    fresh_live_updates=len(fresh),fresh_live_seconds_per_update=seconds,production_benchmark_max_relative_difference=max_relative,production_checks=checks,
    max_action_flow_output_relative_rmse=max(x['student_action_relative_rmse'] for x in speed['numerics']),maximum_task_action_flow_output_relative_rmse=tasks['maximum_task_relative_rmse'],
    minimum_mask_gradient_cosine=min(v for row in speed['numerics'] for k,v in row.items() if k.endswith('_gradient_cosine')),
    teacher_action_flow_outputs_identical=all(x['teacher_action_relative_rmse']==0 for x in speed['numerics']),reliability_rejection_flip_fraction=max(x['reliability_rejection_flip_fraction'] for x in speed['numerics']),
    calibration_max_relative_norm_difference=cal['maximum_relative_difference'],full_calibration_peak_allocated_gib=gpu['peak_allocated_gib'],
    objectives_preserved={'ground_truth_action':1.,'reliable_teacher_video':.05,'reliable_KV_layers_20_to_39':.05},
    stopping_preserved={'earliest_eligible_update':2000,'maximum_update':3000,'original_coverage_stability_and_readiness_required':True},
    remaining_update_hours_to_2000=max(0,2000-latest)*seconds/3600,remaining_update_hours_to_3000=max(0,3000-latest)*seconds/3600,
    eta_excludes_calibration_checkpoint_and_validation_overhead=True,
    wandb='https://wandb.ai/bakery_ismail-bagel-network/vpp2-thaw-robodojo/runs/flexible-mask-'+gpu['identity'][:12],
    evidence=dict(gpu=str(root/'checks/gpu.json'),benchmark=str(prior/'checks/speed_micro4.json'),task_audit=str(prior/'checks/speed_micro4_tasks.json'),calibration=str(root/'checks/calibration_equivalence.json')))
(root/'checks/optimization_summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='production_checks'},indent=2))
