import datetime,json,math,pathlib,subprocess,time
root=pathlib.Path('/data3/praveen/experiments/vpp2-selected-mask-recovery-curated30k-20261101')
identity=json.loads((root/'source_manifest.json').read_text())['identity']
last=None
for attempt in range(45):
    queue=subprocess.run(['squeue','-h','-j','50858','-o','%T %M %R'],capture_output=True,text=True).stdout.strip()
    metric_file=root/'recovery/metrics.jsonl'
    rows=[]
    if metric_file.exists():
        for line in metric_file.read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:pass
    summary={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':50858,'queue':queue,'updates':len(rows)}
    if rows:
        summary.update({k:rows[-1][k] for k in ['update','subset_update','seconds_per_update','gt_action','total','video_gradient_norm']})
    if summary.get('subset_update')!=last or attempt%3==0:
        print(json.dumps(summary),flush=True);last=summary.get('subset_update')
    if len(rows)>=2:
        good=all(row['update']==55+row['subset_update'] and row['subset_update']==i+1 and
            row['subset_windows_processed']==256*(i+1) and
            math.isfinite(row['total']) and math.isfinite(row['video_gradient_norm']) and row['video_gradient_norm']>0 and
            abs(row['late_kv_weight']-.05)<1e-7 and abs(row['teacher_video_weight']-.05)<1e-7 and row['mask_lr']==0 for i,row in enumerate(rows))
        if not good:raise RuntimeError('Unexpected recovery metrics/cursor/loss weights')
        record=dict(summary,identity=identity,passed=True,parent_update=55,phase_updates_planned=2344,
            finite_first_updates=True,loss_weights_unchanged=True,mask_training_disabled=True,
            checkpoint_restore_verified_by_training=True,matching_native_baseline=(root/'recovery/diagnostics/native_000055.json').exists())
        (root/'checks/startup_verified.json').write_text(json.dumps(record,indent=2)+'\n')
        print('STARTUP_VERIFIED '+json.dumps(record),flush=True)
        break
    if not queue:
        print('JOB_LEFT_QUEUE_BEFORE_TWO_UPDATES',flush=True)
        raise SystemExit(2)
    time.sleep(20)
else:
    print('Still initializing or waiting; no successful-start claim made.',flush=True)
