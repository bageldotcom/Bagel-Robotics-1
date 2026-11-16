"""Teacher reliability from disjoint training episodes, never validation targets."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import default_collate
from vpp2.compact_kv.spec import atomic_json,stable_seed,file_sha
from vpp2.compact_kv.model import prepare_batch,native_cache,action_prediction
from vpp2.compact_kv.losses import action_error
from vpp2.width_kv.common import training_dataset
from .common import identity,verify_receipt
from .runtime import initialize,frozen_reference
from .reliability import thresholds_from_rows
from .data_scope import task_scope, validate_training_dataset


def panel_entries(dataset,seed):
    prefix=np.r_[0,np.cumsum(dataset.trajectory_lengths)[:-1]]
    entries=[]
    for task in sorted(dataset.metadata.task_name.unique()):
        indices=np.flatnonzero(dataset.metadata.task_name.to_numpy()==task)
        if len(indices)<80: raise ValueError('Need 80 distinct training episodes per task')
        gen=np.random.default_rng(stable_seed(seed,'teacher-reliability-panel',str(task)))
        for order,trajectory in enumerate(gen.permutation(indices)[:80]):
            length=int(dataset.trajectory_lengths[trajectory])
            last=length-int(dataset.action_horizon)
            if last<0: raise ValueError('Calibration episode has no complete action horizon')
            step=round((order%3)*last/2)
            row=dataset.metadata.iloc[trajectory]
            entries.append(dict(id=len(entries),task=str(task),trajectory=int(trajectory),step=step,
                index=int(prefix[trajectory])+step,episode_index=int(row.episode_index),
                partition='calibration' if order<64 else 'diagnostic'))
    return entries


def prepared_draw(reference,sample,entry,draw,seed):
    position=-10_000_000-(entry['id']*4+draw)
    sample['pilot_position']=torch.tensor([position])
    inputs,noise,tv,_,_,_,mask=prepare_batch(reference,sample,seed)
    action=inputs['action']
    gen=torch.Generator(device=action.device).manual_seed(stable_seed(seed,'reliability-action',position))
    eps=torch.randn(action.shape,device=action.device,dtype=action.dtype,generator=gen)
    u=(draw+torch.rand(1,device=action.device,generator=gen))/4
    scheduler=reference.train_action_scheduler
    ta=(scheduler._phi(u,scheduler.shift)*scheduler.num_train_timesteps).to(action.dtype)
    return inputs,noise,tv,scheduler.add_noise(action,eps,ta),ta,scheduler.training_target(action,eps,ta),mask,u


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    config=json.loads(Path(a.config).read_text());root=Path(config['root']);ident=identity(config)
    verify_receipt(root/'checks/cpu.json',ident)
    if (root/'checks/reliability.json').exists():
        receipt=verify_receipt(root/'checks/reliability.json',ident)
        if file_sha(root/'calibration/thresholds.json')!=receipt['thresholds_sha256']:
            raise RuntimeError('Frozen teacher thresholds changed')
        print('Reusing immutable verified reliability calibration',flush=True)
        return
    rank,world,_,_=initialize(config)
    dataset=training_dataset(config)
    validate_training_dataset(dataset,config)
    tasks=task_scope(config)[0]
    entries=panel_entries(dataset,config['seed'])
    if config['teacher_calibration']!=dict(train_episodes_per_task=64,diagnostic_episodes_per_task=16,draws=4):
        raise RuntimeError('Reliability recipe changed')
    expected_draws=len(tasks)*80*4
    if {e['task'] for e in entries}!=set(tasks):raise RuntimeError('Calibration task set changed')
    if rank==0: atomic_json(root/'calibration/reliability_panel.json',entries)
    reference=frozen_reference(config)
    local=[];start=time.monotonic()
    with torch.no_grad():
        for n,entry in enumerate(entries[rank::world]):
            sample=default_collate([dataset._get(entry['index'])])
            if int(sample['trajectory_idx'][0])!=entry['trajectory'] or int(sample['step_idx'][0])!=entry['step']:
                raise RuntimeError('Calibration sample identity changed')
            for draw in range(4):
                inputs,noise,tv,noisy,ta,target,mask,u=prepared_draw(reference,sample,entry,draw,config['seed'])
                cache=native_cache(reference,inputs,noise,tv,mask)
                prediction=action_prediction(reference,inputs,cache,noisy,ta,mask)
                valid=inputs.get('action_loss_is_pad',inputs.get('action_is_pad'))
                if inputs.get('action_is_pad') is not None and inputs['action_is_pad'].any():
                    raise RuntimeError('Calibration requires a complete ground-truth action horizon')
                error=float(action_error(prediction,target,valid))
                zero=float(action_error(torch.zeros_like(target),target,valid))
                if not np.isfinite([error,zero]).all(): raise FloatingPointError('Nonfinite calibration error')
                local.append(dict(**entry,draw=draw,stratum=draw,u=float(u),teacher_error=error,zero_error=zero))
                del cache,prediction,inputs
            if n%8==0: print(json.dumps(dict(event='reliability_calibration',rank=rank,episodes=n+1,seconds=time.monotonic()-start)),flush=True)
    gathered=[None]*world;dist.all_gather_object(gathered,local)
    if rank==0:
        rows=sorted([row for part in gathered for row in part],key=lambda r:(r['id'],r['draw']))
        if len(rows)!=expected_draws or len({(r['id'],r['draw']) for r in rows})!=expected_draws:
            raise RuntimeError('Incomplete or duplicate teacher calibration draws')
        table=thresholds_from_rows([r for r in rows if r['partition']=='calibration'])
        atomic_json(root/'calibration/reliability_rows.json',rows)
        atomic_json(root/'calibration/thresholds.json',table)
        diagnostics={}
        for task in table:
            group=[r for r in rows if r['task']==task and r['partition']=='diagnostic']
            diagnostics[task]=dict(windows=len(group),teacher_error=float(np.mean([r['teacher_error'] for r in group])),
                fraction_worse_than_zero=float(np.mean([r['teacher_error']>r['zero_error'] for r in group])),
                fraction_above_calibrated_q95=float(np.mean([r['teacher_error']>table[task][r['stratum']]['q95'] for r in group])))
        atomic_json(root/'checks/reliability.json',dict(passed=True,identity=ident,training_tasks=list(tasks),training_draws=len(tasks)*64*4,diagnostic_draws=len(tasks)*16*4,
            thresholds_sha256=file_sha(root/'calibration/thresholds.json'),diagnostics=diagnostics))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
