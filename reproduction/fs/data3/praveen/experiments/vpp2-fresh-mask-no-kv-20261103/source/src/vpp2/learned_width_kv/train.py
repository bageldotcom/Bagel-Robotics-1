"""Two isolated 1000-update phases; complete-loss scaling and FP32 mask optimizer."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from vpp2.compact_kv.model import native_config,prepare_batch,native_cache,action_prediction
from vpp2.compact_kv.losses import kv_terms
from vpp2.compact_kv.sampling import PositionDataset,GlobalBatchSampler
from vpp2.compact_kv.spec import atomic_json,lr_multiplier,stable_seed,file_sha
from vpp2.compact_kv.validation import make_panel
from vpp2.compact_kv.checkpointing import restore_rng,rng_state
from vpp2.compact_kv.gpu_check import tensor_hash
from vpp2.compact_kv.train import assert_student_geometry
from .common import identity,verify_receipt,kv_coefficient
from .runtime import initialize,student_engine,frozen_reference,student_cache
from .controller import synchronize_controller
from .reliability import action_losses,kv_losses
from .validation import evaluate
from .checkpointing import save_training,export_phase
from .reporting import LearnedReporter
from .data_scope import validate_training_dataset, task_scope
from .lifecycle import interruption_needed

STOP=False
def request_stop(*_):
    global STOP
    STOP=True


def fingerprint(value):
    """Device-independent exact tensor/optimizer digest for interrupted-run checks."""
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            x=x.detach().cpu().contiguous();h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--phase',choices=['A','B'],required=True)
    p.add_argument('--micro-batch',type=int);p.add_argument('--trial');p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--resume-from');p.add_argument('--save-at',type=int,default=0)
    a=p.parse_args();config=json.loads(Path(a.config).read_text());root=Path(config['root']);ident=identity(config)
    if not a.trial and a.steps!=1000:raise ValueError('Main phase must contain exactly 1000 optimizer updates')
    verify_receipt(root/'checks/cpu.json',ident)
    rel=verify_receipt(root/'checks/reliability.json',ident)
    thresholds=json.loads((root/'calibration/thresholds.json').read_text())
    if file_sha(root/'calibration/thresholds.json')!=rel['thresholds_sha256']:raise RuntimeError('Teacher thresholds changed')
    if set(thresholds)!=set(task_scope(config)[0]):raise RuntimeError('Teacher thresholds must cover exactly the training tasks')
    if not a.trial:
        verification=verify_receipt(root/f'checks/ready_{a.phase}.json',ident)
        if a.micro_batch is not None and a.micro_batch!=verification['micro_batch']:
            raise RuntimeError('Main execution must use the tested microbatch')
        micro=verification['micro_batch']
    else:
        if a.micro_batch not in config['profile_candidates'][a.phase]: raise ValueError('Unapproved profile microbatch')
        micro=a.micro_batch
    rank,world,local,_=initialize(config)
    for sig in (signal.SIGTERM,signal.SIGUSR1):signal.signal(sig,request_stop)
    directory=root/'profiles'/a.trial if a.trial else root/'runs'/a.phase
    directory.mkdir(parents=True,exist_ok=True)
    if rank==0:(directory/'INTERRUPTED.json').unlink(missing_ok=True)
    selection=json.loads((root/'calibration/initial_selection.json').read_text())
    engine,controller,mask_optimizer=student_engine(config,a.phase,micro,selection)
    start=0;resume_rng=None;seen=set();previous_selection=selection
    if a.phase=='B':
        coverage=json.loads((root/'runs/A/coverage.json').read_text())
        if coverage['identity']!=ident:raise RuntimeError('Phase A coverage identity changed')
        seen={tuple(x) for x in coverage['seen_windows']}
    checkpoint_root=Path(a.resume_from) if a.resume_from else directory/'checkpoints'
    if (checkpoint_root/'latest').exists():
        tag=(checkpoint_root/'latest').read_text().strip()
        receipt=verify_receipt(checkpoint_root/tag/'COMPLETE.json',ident)
        # Every rank verifies its local partition; the complete manifest was committed before latest.
        files=list(sorted(receipt['files']))
        for name in files[rank::world]:
            item=receipt['files'][name];path=checkpoint_root/tag/name
            if path.stat().st_size!=item['size'] or file_sha(path)!=item['sha256']:
                raise RuntimeError('Resumable partition checksum mismatch')
        dist.barrier()
        path,client=engine.load_checkpoint(str(checkpoint_root),tag=tag,load_optimizer_states=True,load_lr_scheduler_states=False)
        if not path or client['identity']!=receipt['identity'] or client['phase']!=a.phase:raise RuntimeError('Resume identity changed')
        if client['thresholds_sha256']!=rel['thresholds_sha256']:raise RuntimeError('Resume teacher thresholds changed')
        if client['extra'].get('micro_batch')!=micro:raise RuntimeError('Resume microbatch changed')
        start=int(client['update'])
        if client['next_global_sample']!=(start+(1000 if a.phase=='B' else 0))*288:raise RuntimeError('Invalid sample cursor')
        if controller is not None:
            controller.load_state_dict(client['controller'],strict=True);mask_optimizer.load_state_dict(client['mask_optimizer'])
        resume_rng=client['rank_rng'][rank];seen={tuple(x) for x in client['extra'].get('seen_windows',[])}
        previous_selection=client['extra'].get('previous_selection',selection)
        if engine.global_steps!=start:raise RuntimeError('DeepSpeed update counter not restored')
    if not 0<=start<=a.steps or (a.trial and start==a.steps):raise ValueError('Invalid phase restart endpoint')
    reference=frozen_reference(config,micro)
    action_sha=tensor_hash(reference.action_expert) if rank==0 else None
    frozen_digest=fingerprint({n:getattr(p,'ds_tensor',p) for n,p in reference.named_parameters()})
    native=instantiate(native_config(config).data.train)
    validate_training_dataset(native,config)
    dataset=PositionDataset(native,config['seed']);offset=1000 if a.phase=='B' else 0
    sampler=GlobalBatchSampler(start+offset,a.steps+offset,rank,world,micro,288)
    loader=DataLoader(dataset,batch_sampler=sampler,num_workers=config['num_workers'],pin_memory=True,
        persistent_workers=config['num_workers']>0,generator=torch.Generator().manual_seed(stable_seed(config['seed'],'loader',rank)),
        **(dict(prefetch_factor=2,multiprocessing_context='spawn') if config['num_workers'] else {}))
    panel=make_panel(config)
    reporter=LearnedReporter(config) if rank==0 and not a.trial else None
    def report(row):
        if rank!=0:return
        with (directory/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
        if reporter:reporter.write(row)
        print(json.dumps({k:v for k,v in row.items() if not isinstance(v,(dict,list))}),flush=True)
    if start==0 and not a.trial:
        row=evaluate(engine,controller,reference,panel,config,a.phase,0,directory,thresholds,full=True,mode='open' if a.phase=='A' else 'soft')
        if rank==0:report(row)
        if a.phase=='B':
            export_phase(engine,controller,config,a.phase,0,ident)
            from .verification import verify_native_export
            verify_native_export(config,0,reference,ident)
    # A process can fail after a valid checkpoint but before a milestone export
    # or the final phase receipt. Repair those outputs from the restored model.
    if start and not a.trial and start in (250,500,750,1000):
        row=evaluate(engine,controller,reference,panel,config,a.phase,start,directory,thresholds,full=True)
        if rank==0:report(row)
        if a.phase=='A':
            row=evaluate(engine,controller,reference,panel,config,a.phase,start,directory,thresholds,full=True,mode='binary')
            if rank==0:report(row)
        if a.phase=='B' or start==1000:
            export_phase(engine,controller,config,a.phase,start,ident)
            if a.phase=='B':
                from .verification import verify_native_export
                verify_native_export(config,start,reference,ident)
    engine.train();reference.mot.training=True
    if resume_rng is not None:restore_rng(resume_rng)
    accumulation=sampler.accumulation;update=start;microstep=0;records=[];identities=[]
    start_time=time.monotonic();step_start=start_time;torch.cuda.reset_peak_memory_stats()
    duration=[];stopped=False
    for sample in loader:
        next_update=update+1
        fraction=lr_multiplier(update,1000,50,.1)
        lr=(config['mask_phase_video_lr'] if a.phase=='A' else config['recovery_video_lr'])*fraction
        for group in engine.optimizer.param_groups:group['lr']=lr
        if mask_optimizer:
            for group in mask_optimizer.param_groups:group['lr']=config['mask_lr']*fraction
        coefficient=kv_coefficient(config,a.phase,next_update)
        inputs,noise,tv,noisy,ta,target,mask,u=prepare_batch(reference,sample,config['seed'],return_noise_u=True)
        tasks=[str(native.metadata.iloc[int(i)].task_name) for i in sample['trajectory_idx']]
        strata=(u*4).long().clamp_max(3).tolist()
        with torch.no_grad():
            teacher=native_cache(reference,inputs,noise,tv,mask)
            teacher_action=action_prediction(reference,inputs,teacher,noisy,ta,mask)
        student,gates=student_cache(engine,controller,a.phase,next_update,inputs,noise,tv,mask)
        assert_student_geometry(student,teacher)
        prediction=action_prediction(reference,inputs,student,noisy,ta,mask)
        valid=inputs.get('action_loss_is_pad',inputs.get('action_is_pad'))
        tw=reference.train_action_scheduler.training_weight(ta)
        gt,kd,raw,weights,reasons=action_losses(prediction,teacher_action,target,valid,tw,tasks,strata,thresholds)
        terms=kv_terms(student,teacher)
        kv,kv_gated,kv_gated_layers=kv_losses(terms,weights)
        regularizer=kv.new_zeros(());penalties={}
        if controller:regularizer,penalties=controller.regularizers(next_update,gates)
        loss=gt+kd+coefficient*kv_gated+regularizer
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training objective')
        # Disable DeepSpeed's output-hook scaling. It cannot scale an external
        # controller-only regularizer; scale the entire graph exactly once here.
        engine.backward(loss/accumulation,scale_wrt_gas=False)
        if any(p.grad is not None for p in reference.parameters()):raise RuntimeError('Frozen native parameter acquired a gradient')
        microstep+=1;boundary=microstep%accumulation==0
        mask_norm=0.
        if boundary and controller:
            mask_norm=float(synchronize_controller(controller,1.))
        engine.step()
        if boundary and mask_optimizer:
            mask_optimizer.step();mask_optimizer.zero_grad(set_to_none=True)
        records.append(torch.cat([torch.stack([gt,raw,kd,kv,coefficient*kv_gated,loss,weights.mean(),regularizer,
            reasons['reliability'].mean(),reasons['better_student'].float().mean(),reasons['worse_than_zero'].float().mean(),
            *[penalties.get(k,kv.new_zeros(())) for k in ('count','entropy','discreteness')],
            kv_gated,(weights==0).float().mean()]).detach(),
            terms.detach().mean(0).reshape(-1),kv_gated_layers.reshape(-1)]).double())
        identities.extend((int(pos),int(traj),int(step)) for pos,traj,step in zip(sample['pilot_position'],sample['trajectory_idx'],sample['step_idx']))
        del student,teacher,prediction,teacher_action,terms,loss,inputs,gates,gt,kd,raw,kv,kv_gated,kv_gated_layers,regularizer,penalties
        if not boundary:continue
        update+=1
        if engine.global_steps!=update:raise RuntimeError('Optimizer counter and phase update diverged')
        video_norm=float(engine.get_global_grad_norm())
        if not np.isfinite([video_norm,mask_norm]).all():raise FloatingPointError('Nonfinite gradients')
        values=torch.stack(records).mean(0);dist.all_reduce(values);values/=world;records.clear()
        all_ids=[None]*world;dist.all_gather_object(all_ids,identities)
        ordered=sorted(x for part in all_ids for x in part);identities.clear()
        if [r[0] for r in ordered]!=list(range((offset+update-1)*288,(offset+update)*288)):
            raise RuntimeError('Global sample sequence changed with microbatch layout')
        seen.update((r[1],r[2]) for r in ordered)
        torch.cuda.synchronize();elapsed=time.monotonic()-step_start;duration.append(elapsed)
        peaks=torch.tensor([torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()],device=f'cuda:{local}',dtype=torch.float64)
        dist.all_reduce(peaks,op=dist.ReduceOp.MAX)
        v=values.cpu().tolist()
        row=dict(kind='train',phase=a.phase,phase_update=update,update=offset+update,action_gt=v[0],action_teacher=v[1],
            action_teacher_weighted=v[2],kv_raw=v[3],kv_weighted=v[4],loss=v[5],teacher_action_weight=v[6],mask_regularizer=v[7],
            teacher_reliability=v[8],fraction_teacher_worse_than_student=v[9],fraction_teacher_worse_than_zero=v[10],
            mask_count=v[11],mask_entropy=v[12],mask_discreteness=v[13],kv_gated=v[14],
            teacher_supervision_rejected_fraction=v[15],teacher_kv_weight=v[6],kv_effective_weight=coefficient*v[6],
            kv_by_layer_kv=np.array(v[16:96]).reshape(40,2).tolist(),
            kv_gated_by_layer_kv=np.array(v[96:176]).reshape(40,2).tolist(),
            learning_rate=lr,kv_weight=coefficient,video_gradient_norm=video_norm,mask_gradient_norm=mask_norm,
            seconds_per_update=elapsed,windows_per_second=288/elapsed,global_batch=288,micro_batch=micro,
            max_gpu_allocated_GiB=float(peaks[0])/2**30,max_gpu_reserved_GiB=float(peaks[1])/2**30,
            unique_windows=len(seen),sample_sequence_sha256=fingerprint(ordered),global_windows=(offset+update)*288)
        if controller:
            row.update(controller.diagnostics(update))
            chosen=controller.selection()
            row['hidden_topk_turnover']=1-len(set(chosen['hidden_indices'])&set(previous_selection['hidden_indices']))/2048
            row['ffn_topk_turnover']=sum(1-len(set(x)&set(y))/8192 for x,y in zip(chosen['ffn_indices'][:-1],previous_selection['ffn_indices'][:-1]))/39
            previous_selection=chosen
        report(row)
        stopped_tensor=torch.tensor([int(STOP or (not a.trial and time.monotonic()-start_time>42*3600))],device=f'cuda:{local}')
        dist.all_reduce(stopped_tensor,op=dist.ReduceOp.MAX)
        stopped=interruption_needed(bool(stopped_tensor.item()),update,a.steps)
        full=update in (250,500,750,1000)
        if not a.trial and (full or update%100==0 or stopped):
            metric=evaluate(engine,controller,reference,panel,config,a.phase,update,directory,thresholds,full=full or stopped)
            if rank==0:report(metric)
            if a.phase=='A' and full:
                metric=evaluate(engine,controller,reference,panel,config,a.phase,update,directory,thresholds,full=True,mode='binary')
                if rank==0:report(metric)
        if not a.trial and ((a.phase=='A' and update==1000) or (a.phase=='B' and full)):
            export_phase(engine,controller,config,a.phase,update,ident)
            if a.phase=='B':
                from .verification import verify_native_export
                verify_native_export(config,update,reference,ident)
        # Commit resumable training state only after this milestone's exports
        # and reload checks. A final-checkpoint restart can finalize idempotently.
        if (not a.trial and (update%250==0 or stopped)) or update==a.save_at:
            save_training(engine,controller,mask_optimizer,directory,a.phase,update,config,ident,
                extra=dict(seen_windows=sorted(seen),previous_selection=previous_selection,micro_batch=micro))
        if stopped:break
        gc.collect();step_start=time.monotonic()
    if rank==0 and tensor_hash(reference.action_expert)!=action_sha:raise RuntimeError('Frozen action weights changed')
    if fingerprint({n:getattr(p,'ds_tensor',p) for n,p in reference.named_parameters()})!=frozen_digest:
        raise RuntimeError('Frozen teacher, encoder or action weights changed')
    if not a.trial and rank==0:
        atomic_json(directory/'coverage.json',dict(identity=ident,phase=a.phase,update=update,seen_windows=sorted(seen)))
    if a.trial:
        digest=dict(model=fingerprint({n:getattr(p,'ds_tensor',p) for n,p in engine.module.named_parameters()}),
            optimizer=fingerprint(engine.optimizer.state_dict()),
            controller=None if controller is None else fingerprint(controller.state_dict()),
            mask_optimizer=None if mask_optimizer is None else fingerprint(mask_optimizer.state_dict()),rng=fingerprint(rng_state()))
        digests=[None]*world;dist.all_gather_object(digests,digest)
        if rank==0:
            atomic_json(directory/'COMPLETE.json',dict(passed=True,identity=ident,phase=a.phase,update=update,micro_batch=micro,
                durations=duration,mean_steady_seconds=float(np.mean(duration[2:] if len(duration)>2 else duration)),
                max_reserved_gib=float(peaks[1])/2**30,digests=digests,action_unchanged=True))
    elif not stopped:
        if rank==0:atomic_json(directory/'COMPLETE.json',dict(passed=True,identity=ident,phase=a.phase,update=update,global_update=offset+update,action_unchanged=True))
    if stopped and rank==0:
        atomic_json(directory/'INTERRUPTED.json',dict(passed=True,identity=ident,phase=a.phase,update=update,
            job_id=os.environ.get('SLURM_JOB_ID'),checkpoint=str(directory/'checkpoints'/f'update_{update:06d}')))
    if reporter:reporter.close()
    dist.barrier();dist.destroy_process_group()
    if stopped:sys.exit(75)


if __name__=='__main__':main()
