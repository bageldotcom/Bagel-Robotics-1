"""Single joint selection trajectory. No capacity retries, rollback or job submission."""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import signal
import time
import torch
import torch.distributed as dist
from vpp2.compact_kv.spec import atomic_json
from . import checkpoint
from .coefficient import Coefficient
from .config import load,identity
from .controller import schedule
from .data import stream,positions,coverage
from .metrics import DeferredMetrics
from .selection import readiness,candidate_key

def control_stop(root):
    path=Path(root)/'control/stop_request.json'
    if not path.exists():return False
    request=json.loads(path.read_text())
    return request.get('job_id')==os.environ.get('SLURM_JOB_ID')


STOP=False
def request_stop(*_):
    global STOP
    STOP=True


def learning_rate_fraction(update,total,warmup,final=.1):
    if update<=warmup:return update/warmup
    return final+(1-final)*.5*(1+math.cos(math.pi*min((update-warmup)/(total-warmup),1)))


def mean_distributed(tensor):
    dist.all_reduce(tensor);tensor/=dist.get_world_size()
    return tensor


def run(config,stage='mask',recovery_export=None,smoke_updates=None,probe=False,probe_name='default'):
    from .runtime import Runtime,distributed
    from .diagnose import native,flow
    from .preflight import require_receipt
    from .report import Reporter
    rank,world,_=distributed(config)
    ident=identity(config);root=Path(config['root'])
    require_receipt(config,'cpu')
    if not probe:
        if config.get('pressure_continuation'):
            from .pressure import require_runtime_evidence
            require_runtime_evidence(config)
        else:
            require_receipt(config,'gpu')
        if config.get('execution',{}).get('teacher_storage')=='replicated':require_receipt(config,'performance')
    if stage not in ('mask','recovery'):raise ValueError(stage)
    if not probe_name.replace('_','').isalnum():raise ValueError('Invalid probe name')
    directory=root/(('gpu_probe_'+probe_name) if probe else stage);directory.mkdir(parents=True,exist_ok=True)
    (directory/'diagnostics').mkdir(exist_ok=True)
    runtime=Runtime(config,recovery_export=recovery_export)
    panel=json.loads(Path(config['panel_file']).read_text());samples=stream(panel,config['seed'],stage=='recovery')
    state=checkpoint.load(runtime,directory,ident,stage)
    if state is None and not probe and config.get('resume_from'):
        from .continuation import load_parent
        state=load_parent(runtime,config,stage)
    update=0 if state is None else state['update']
    coefficient=(Coefficient(config['size_coefficient']['upper_cap']) if state is None else
                 Coefficient.from_state(state['coefficient'],config['size_coefficient']['upper_cap']))
    if config.get('pressure_continuation'):
        if stage!='mask' or probe:raise ValueError('Pressure continuation requires checkpointed mask training')
        from .pressure import verify_gpu
        verify_gpu(runtime,config,state,coefficient)
    history=[] if state is None else state['history'];best=[] if state is None else state['best']
    max_updates=3000 if stage=='mask' else 6000
    endpoint=smoke_updates if smoke_updates is not None else max_updates
    if state is not None and not probe:
        stopped_status=None
        if coefficient.unresolved:stopped_status=dict(status='coefficient_unresolved',update=update)
        elif stage=='recovery' and update==6000:stopped_status=dict(status='recovery_complete',update=update)
        elif stage=='mask' and history and history[-1]['update']==update:
            status=history[-1].get('readiness')
            if status and (status['status']=='ready_for_recovery' or update==3000):stopped_status=status
        if stopped_status:
            if rank==0:atomic_json(directory/'status.json',dict(stopped_status,identity=ident,jobs_submitted=False))
            dist.barrier()
            return runtime,stopped_status
    if not update<endpoint<=max_updates:raise ValueError('Invalid training endpoint')
    if smoke_updates is not None and not probe:raise ValueError('Short runs are only verification probes')
    for sig in (signal.SIGUSR1,signal.SIGTERM):signal.signal(sig,request_stop)
    reporter=Reporter(config,directory,stage,enabled=not probe) if rank==0 else None
    previous_scores=None if runtime.masks is None else ([p.detach().clone() for p in runtime.masks.families()]
        if state is None else [p.to(runtime.device) for p in state['previous_scores']])
    interval_grad=torch.zeros(3,device=runtime.device) if state is None else state['interval_grad'].to(runtime.device)
    interval_n=0 if state is None else state['interval_n']
    final_status=None
    begin=time.monotonic()
    frozen_initial=None
    # Fingerprints compare immutable partition contents, not ZeRO parameter _version.
    from .preflight import frozen_fingerprint
    frozen_initial=frozen_fingerprint(runtime)
    if update==0 and not probe:
        initial=native(runtime,panel,0)
        if rank==0:atomic_json(directory/'diagnostics/native_000000.json',initial)
    if probe and update==0:
        from .preflight import gpu_contracts
        contracts=gpu_contracts(runtime,panel)
        from .preflight import runtime_fingerprint
        atomic_json(directory/f'initial_fingerprint_rank{rank}.json',dict(identity=ident,fingerprint=runtime_fingerprint(runtime)))
        if rank==0:atomic_json(directory/'contracts.json',dict(contracts,identity=ident))
    runtime.engine.train()
    for u in range(update+1,endpoint+1):
        tick=time.monotonic();collected=DeferredMetrics(runtime.device)
        warm=50 if stage=='mask' else 250
        fraction=learning_rate_fraction(u,max_updates,warm)
        video_lr=(1e-6 if stage=='mask' else 5e-6)*fraction
        for group in runtime.engine.optimizer.param_groups:group['lr']=video_lr
        if runtime.mask_optimizer:
            for group in runtime.mask_optimizer.param_groups:group['lr']=.01*fraction
        batches=positions(u,rank,world,config['train']['micro_batch']);accum=len(batches)
        measure=runtime.masks is not None and (coefficient.collecting(u) or (probe and probe_name=='gradient_probe'))
        gt_vectors=[torch.zeros_like(p) for p in runtime.masks.families()] if measure else None
        size_vectors=None
        auxiliary_vectors={k:[torch.zeros_like(p) for p in runtime.masks.families()] for k in ('video','kv')} if measure else None
        from .pressure import weight as pressure_weight
        weight=pressure_weight(coefficient,config,u) if runtime.masks else 0.
        for pos in batches:
            rows=[samples.at(p) for p in pos]
            loss,metrics,details,terms=runtime.forward(rows,pos,u,measure_gt=measure)
            if runtime.masks is not None:
                size=runtime.cost.loss(runtime.masks.gates(u))
                entropy,discrete=runtime.masks.regularizers(u);s=schedule(u)
                if measure:
                    # A separate module-only pass avoids DeepSpeed's optimizer
                    # backward hooks. Its weights/inputs are identical to this pass.
                    for out,g in zip(gt_vectors,runtime.measured_mask_gradients['gt']):out.add_(g/accum)
                    for key in auxiliary_vectors:
                        for out,g in zip(auxiliary_vectors[key],runtime.measured_mask_gradients[key]):out.add_(g/accum)
                    if size_vectors is None:
                        size_vectors=[g.detach() for g in torch.autograd.grad(size,runtime.masks.families(),retain_graph=True)]
                loss=loss+weight*size+s['entropy']*entropy+s['discreteness']*discrete
                metrics.update(size_fraction=size.detach(),size_weight=weight,entropy=entropy.detach(),
                    discreteness=discrete.detach(),entropy_weight=s['entropy'],discreteness_weight=s['discreteness'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
            # DeepSpeed scales returned outputs by accumulation by default. Disable
            # that route and scale the COMPLETE loss exactly once, including masks.
            if measure:
                # NCCL allocates outside PyTorch's caching allocator. Release
                # unused diagnostic blocks before gradient communication; live
                # tensors, arithmetic and optimizer accumulation are untouched.
                from .calibration_batch import release_unused_cache
                released=release_unused_cache(runtime.device)
                metrics['calibration_allocator_release_gib']=released['released_gib']
            runtime.engine.backward(loss/accum,scale_wrt_gas=False)
            runtime.check_frozen()
            boundary=runtime.engine.is_gradient_accumulation_boundary()
            if boundary and runtime.masks is not None:
                for p in runtime.masks.families():
                    if p.grad is None or not torch.isfinite(p.grad).all():raise FloatingPointError('Missing/nonfinite mask gradient')
                    mean_distributed(p.grad)
                norms=torch.stack([p.grad.norm() for p in runtime.masks.families()])
                interval_grad+=norms;interval_n+=1
                torch.nn.utils.clip_grad_norm_(runtime.masks.parameters(),1.,error_if_nonfinite=True)
            runtime.engine.step()
            if boundary and runtime.mask_optimizer:
                runtime.mask_optimizer.step();runtime.mask_optimizer.zero_grad(set_to_none=True)
            metrics['total']=loss.detach()
            collected.append(metrics,details,rows)
            del loss,terms,metrics,details
        if runtime.engine.global_steps!=u:raise RuntimeError('Accumulation/update counter mismatch')
        gradient_report={}
        if measure:
            gn=[float(mean_distributed(v).norm()) for v in gt_vectors]
            sn=[float(mean_distributed(v).norm()) for v in size_vectors]
            if coefficient.collecting(u):coefficient.observe(u,gn,sn)
            auxiliary_norms={k:[float(mean_distributed(v).norm()) for v in vs] for k,vs in auxiliary_vectors.items()}
            for i,family in enumerate(('hidden','ffn','cross')):
                gradient_report['calibration/'+family+'/gt_norm']=gn[i]
                gradient_report['calibration/'+family+'/size_norm']=sn[i]
                for key in auxiliary_norms:
                    gradient_report['calibration/'+family+'/'+key+'_weighted_norm']=.05*schedule(u)['teacher']*auxiliary_norms[key][i]
        if not math.isfinite(float(runtime.engine.get_global_grad_norm())):raise FloatingPointError('Invalid video gradient norm')
        row,task_metrics=collected.finish()
        keys=sorted(row);vals=torch.tensor([row[k] for k in keys],device=runtime.device,dtype=torch.float64)
        vals=mean_distributed(vals).cpu().tolist();row=dict(zip(keys,vals))
        parts=[None]*world;dist.all_gather_object(parts,task_metrics)
        grouped=defaultdict(list)
        for item in sum(parts,[]):
            grouped['task/'+item['task']].append(item);grouped['suite/'+item['suite']].append(item)
        for key,values in grouped.items():
            for metric in ('student_error','teacher_error','reliability'):
                row[key+'/'+metric]=sum(v[metric] for v in values)/len(values)
        torch.cuda.synchronize();elapsed=time.monotonic()-tick
        row.update(update=u,stage=stage,seconds_per_update=elapsed,windows_per_second=256/elapsed,
                   video_lr=video_lr,mask_lr=.01*fraction if runtime.masks else 0.,
                   video_gradient_norm=float(runtime.engine.get_global_grad_norm()),
                   max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                   coefficient_measured=coefficient.measured,coefficient_capped=coefficient.applied,
                   coefficient_upper_cap=coefficient.cap,
                   coefficient_cap_active=coefficient.measured is not None and coefficient.measured>coefficient.cap)
        if runtime.masks is not None:
            for name,value in zip(('hidden','ffn','cross'),norms):row['gradient/'+name]=float(value)
        row.update(gradient_report)
        if config.get('pressure_continuation'):
            row.update(size_pressure_base=coefficient.weight(u),
                size_pressure_multiplier=weight/coefficient.weight(u),
                size_pressure_target=config['pressure_continuation']['absolute_cap'],
                size_pressure_ramp_complete=u>=config['pressure_continuation']['start_update']+250)
        if rank==0:reporter.write(row)
        stop=torch.tensor(int(STOP or control_stop(root) or time.monotonic()-begin>42*3600),device=runtime.device)
        dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        unresolved=coefficient.unresolved and stage=='mask'
        check=(u%250==0 and not probe)
        if check:
            result=native(runtime,panel,u)
            if rank==0:
                atomic_json(directory/f'diagnostics/native_{u:06d}.json',result)
                reporter.native(result)
            if u%500==0:
                flow_result=flow(runtime,panel,u)
                if rank==0:atomic_json(directory/f'diagnostics/flow_{u:06d}.json',flow_result)
            if runtime.masks:
                selection,added=runtime.masks.selection();parameters=runtime.inventory.count(selection)
                probs=runtime.masks.probabilities(u)
                moved=[float((p.detach()-old).square().mean().sqrt()) for p,old in zip(runtime.masks.families(),previous_scores)]
                active=all(float(n)>1e-12 for n in interval_grad/max(interval_n,1)) and all(x>1e-12 for x in moved)
                active=active and all(float((schedule(u)['temperature']*p*(1-p)).max())>1e-8 for p in probs)
                hist=dict(update=u,selection=selection,parameters=parameters,
                    regularizer_state=[schedule(u)['temperature'],schedule(u)['entropy'],schedule(u)['discreteness'],weight],
                    gradient_active=active,raw_logit_rms_change=moved,actions=result['actions'],restored_channels=added)
                history.append(hist)
                covered=coverage(samples,u*256)
                final_status=readiness(history,runtime.geometry,u,min(v['minimum_visits'] for v in covered.values()))
                hist['readiness']=final_status
                ranked=dict(hist,full_native_actions=hist['actions'])
                best=sorted(best+[ranked],key=candidate_key)[:2]
                if rank==0:reporter.masks(runtime,u,covered,hist)
                previous_scores=[p.detach().clone() for p in runtime.masks.families()]
                interval_grad.zero_();interval_n=0
            elif u%500==0:
                candidate=dict(update=u,parameters=runtime.physical_parameters,full_native_actions=result['actions'])
                best=sorted(best+[candidate],key=candidate_key)[:2]
        ready=final_status is not None and final_status['status']=='ready_for_recovery'
        save_due=check and (stage=='mask' or u%500==0)
        if save_due or stop or unresolved or ready or u==endpoint:
            if frozen_fingerprint(runtime)!=frozen_initial:raise RuntimeError('Frozen teacher/action/encoder parameters changed')
            state=dict(identity=ident,update=u,stage=stage,coefficient=coefficient.state_dict(),history=history,best=best,
                       previous_scores=previous_scores,interval_grad=interval_grad,interval_n=interval_n)
            checkpoint.save(runtime,directory,state)
            if rank==0:
                checkpoint.retain(directory,[f"update_{x['update']:06d}" for x in best])
        if unresolved:
            final_status=dict(status='coefficient_unresolved',update=u);break
        if ready:break
        if stop:
            final_status=dict(status='checkpointed_interruption',update=u);break
    if final_status is None or (stage=='recovery' and u==6000):
        final_status=dict(status='recovery_complete' if stage=='recovery' and u==6000 else 'verification_probe_complete' if probe else 'selection_not_converged',update=u)
    if rank==0:
        atomic_json(directory/'status.json',dict(final_status,identity=ident,jobs_submitted=False))
        reporter.close()
    if probe:
        from .preflight import runtime_fingerprint
        atomic_json(directory/f'fingerprint_rank{rank}.json',dict(identity=ident,update=u,fingerprint=runtime_fingerprint(runtime),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
    dist.barrier()
    return runtime,final_status


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    run(load(a.config))

if __name__=='__main__':main()
