"""Explicit, resumable training-only teacher reliability calibration."""
from collections import Counter,defaultdict
import json
from pathlib import Path
from .panel import digest,immutable
from vpp2.learned_width_kv.reliability import thresholds_from_rows


def sampling_plan(panel):
    grouped=defaultdict(list)
    for row in panel['windows']:
        if row['role']=='selection_initial':grouped[(row['task'],row['episode'])].append(row)
    selected=[]
    for _,rows in sorted(grouped.items()):
        rows=sorted(rows,key=lambda r:r['frame'])
        selected.extend(rows[:1] if len(rows)==1 else [rows[0],rows[-1]])
    counts=Counter((r['task'],s) for r in selected for s in range(4))
    if set(counts)!={(t,s) for t in panel['tasks'] for s in range(4)} or min(counts.values())<64:
        raise ValueError('Teacher calibration requires at least 64 planned training windows per task/noise group')
    return selected,dict(counts)


def validate_rows(rows,selected,counts,panel):
    expected={(r['id'],s) for r in selected for s in range(4)}
    keys=[(r['window_id'],r['stratum']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:
        raise ValueError('Missing, duplicated, or unexpected calibration draws')
    metadata={r['id']:r for r in selected}
    for r in rows:
        source=metadata[r['window_id']]
        if r['task']!=source['task'] or r['episode']!=source['episode']:
            raise ValueError('Calibration window identity mismatch')
    return thresholds_from_rows(rows,expected_counts=counts,expected_tasks=panel['tasks'])


def run(config,model,branch,reader,panel,rank,world):
    import torch
    from .data import collate,prepared
    from .runtime import cache,predict,gather,barrier
    from .train import identity
    from vpp2.compact_kv.losses import action_error
    selected,counts=sampling_plan(panel)
    ident=digest(dict(experiment=identity(config,panel),windows=[r['id'] for r in selected],noise_draws=4))
    path=Path(config['teacher_threshold_file'])
    if path.exists():
        saved=json.loads(path.read_text())
        if saved.get('calibration_identity')!=ident or saved['panel_identity']!=panel['identity']:
            raise ValueError('Threshold provenance mismatch')
        result=validate_rows(saved['rows'],selected,counts,panel)
        if result!=saved['thresholds']:raise ValueError('Stored thresholds differ from their measurements')
        return result
    shard_root=Path(config['root'])/'calibration'/ident
    shard_root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,row in enumerate(selected):
        for stratum in range(4):
            if (i*4+stratum)%world!=rank:continue
            shard=shard_root/f"{row['id']:08d}_{stratum}.json"
            if shard.exists():
                result=json.loads(shard.read_text())
                if result['identity']!=ident:raise ValueError('Stale calibration shard')
                result=result['row']
            else:
                position=-1_000_000-row['id']*4-stratum
                parts=prepared(model,collate(reader,[row],[position]),config['seed'],[stratum])
                with torch.no_grad():
                    teacher=predict(model,parts,cache(branch,parts))
                    error=action_error(teacher,parts[5],parts[0]['action_loss_is_pad'])
                result=dict(window_id=row['id'],task=row['task'],episode=row['episode'],stratum=stratum,teacher_error=float(error))
                immutable(shard,dict(identity=ident,row=result))
            rows.append(result)
            if len(rows)%64==0:print(json.dumps(dict(event='teacher_calibration',rank=rank,local_done=len(rows),total_draws=len(selected)*4)),flush=True)
    rows=sorted(sum(gather(rows),[]),key=lambda r:(r['window_id'],r['stratum']))
    thresholds=validate_rows(rows,selected,counts,panel)
    if rank==0:
        immutable(path,dict(calibration_identity=ident,panel_identity=panel['identity'],thresholds=thresholds,rows=rows,
            expected_counts={t:[counts[(t,s)] for s in range(4)] for t in panel['tasks']},
            scope='First/last selected windows of every fitting episode; four action-noise strata; no monitoring or audit labels.'))
    barrier()
    return thresholds
