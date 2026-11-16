"""Frozen-weight mask search with binary action-quality selection and strict provenance."""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
import torch.distributed as dist
from .panel import BalancedStream,digest,file_hash,immutable,noise_stratum
from .data import collate,prepared
from .runtime import (distributed,barrier,initialize,install,model_versions,check_frozen,
                      step_loss,calibrate_teacher,verify_gpu,gather,verify_checkpoint_receipt)
from .monitor import evaluate,compare,confirm_full
from .native_monitor import evaluate_native
from .search_quality import schedule_end,stable_history,better
from vpp2.learned_width_kv.controller import synchronize_controller

STOP=False

def request_stop(*_):
    global STOP
    STOP=True


def scheduler_stop_requested(root):
    path=Path(root)/'control/stop_request.json'
    if not path.exists():return False
    request=json.loads(path.read_text())
    return str(request.get('job_id'))==os.environ.get('SLURM_JOB_ID','unallocated')


def save(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial');torch.save(state,tmp);tmp.replace(path)


def identity(config,panel):
    source=Path(__file__).resolve().parents[3]
    return digest(dict(config=config,panel=panel['identity'],
        source={str(p.relative_to(source)):file_hash(p) for p in sorted((source/'src/vpp2').rglob('*.py'))}))


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'])


def finalize_selection(root,ident,best_native,best_flow,current,converged,cap):
    if best_native is None:raise RuntimeError('No native-action-audited binary checkpoint; cannot publish a final selection')
    path=Path(best_native['checkpoint'])
    if file_hash(path)!=best_native['checkpoint_sha256']:raise ValueError('Best checkpoint checksum changed')
    selected=torch.load(path,map_location='cpu',weights_only=False)
    if selected['identity']!=ident or selected['update']!=best_native['update']:raise ValueError('Selected checkpoint provenance mismatch')
    # Export indices directly from the saved controller using its recorded config.
    from .controller import Controller
    export=Controller(selected['selection_config']);export.load_state_dict(selected['controller'])
    immutable(root/'runs/final_selection.json',dict(**export.selection(),selected_update=best_native['update'],
        checkpoint_sha256=best_native['checkpoint_sha256'],selection_rule='Lowest equal-task native generated-action MSE ratio; worst-task ratio breaks exact ties.',
        native_quality=best_native['quality'],experiment_identity=ident,selected_mask_stability_confirmed=bool(selected.get('converged',False)),policy_retention_established=False))
    immutable(root/'control/mask_search_finished.json',dict(update=current,selected_update=best_native['update'],converged=converged,
        exhausted_cap=current==cap,interrupted=False,physical_pruning_launched=False,
        best_native=best_native,best_flow=best_flow,selected_mask_stability_confirmed=bool(selected.get('converged',False)),policy_retention_established=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--preflight',action='store_true')
    a=p.parse_args();config=json.loads(Path(a.config).read_text());root=Path(config['root'])
    panel=json.loads(Path(config['panel_file']).read_text())
    if digest({k:v for k,v in panel.items() if k!='identity'})!=panel['identity']:raise ValueError('Panel mutated')
    rank,world=distributed()
    if a.preflight:
        if world!=1:raise ValueError('GPU algebra preflight runs on one allocated GPU')
        verify_gpu(config,panel,rank,world);return
    if 'curation' in config:
        from .verify_data import require_training_readiness
        require_training_readiness(config,panel)
    ident=identity(config,panel)
    gpu=json.loads(Path(config['gpu_receipt_file']).read_text());cpu=json.loads(Path(config['cpu_receipt_file']).read_text())
    if not gpu['passed'] or gpu.get('experiment_identity')!=ident or gpu['panel_identity']!=panel['identity'] or cpu['identity']!=ident or not cpu['passed']:
        raise ValueError('Missing compatible source/configuration/panel readiness checks')
    if world!=8 or config['train']['micro_batch']!=1:raise ValueError('Requires eight allocated GPUs and verified microbatch one')
    distributed_receipt=json.loads(Path(config['distributed_receipt_file']).read_text())
    if distributed_receipt.get('experiment_identity')!=ident or not distributed_receipt.get('passed') or not distributed_receipt.get('resume_bitwise_equal'):
        raise ValueError('Missing compatible eight-GPU smoke/resume validation')
    if rank==0:
        for name in ('runs','control','calibration'): (root/name).mkdir(parents=True,exist_ok=True)
    barrier()
    verify_checkpoint_receipt(config,gpu,rank)
    model,reader,controller=initialize(config);branch=install(model);versions=model_versions(model)
    cfg=config['train'];batch=cfg['global_batch'];end=schedule_end(config)
    if batch%world:raise ValueError('Unbalanced accumulation')
    accumulation=batch//world
    optimizer=torch.optim.AdamW(controller.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['betas']),weight_decay=0.)
    latest=root/'runs/latest.pt';start=0;history=[];previous=None;previous_full=None;best_flow=None;best_native=None;resumed_rng=None
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        if state['identity']!=ident:raise ValueError('Resume identity differs')
        if state['world_size']!=world:raise ValueError('Resume world size differs from audited execution')
        controller.load_state_dict(state['controller']);optimizer.load_state_dict(state['optimizer'])
        start=state['update'];history=state['history'];previous=state['previous'];previous_full=state['previous_full']
        best_flow=state['best_flow'];best_native=state['best_native'];resumed_rng=state['rng'][rank]
        if file_hash(Path(config['teacher_threshold_file']))!=state['thresholds_sha256']:raise ValueError('Resume teacher thresholds changed')
        for selected in (best_flow,best_native):
            if selected is not None and file_hash(Path(selected['checkpoint']))!=selected['checkpoint_sha256']:
                raise ValueError('Selected checkpoint changed since resume')
    if start>=cfg['max_updates'] or (start and state.get('converged',False)):
        if rank==0:
            finalize_selection(root,ident,best_native,best_flow,start,state.get('converged',False),cfg['max_updates'])
            print('Optimizer work complete; final selection publication verified.')
        barrier();return
    thresholds=calibrate_teacher(config,model,branch,reader,panel,rank,world)
    if resumed_rng is not None:restore_rng(resumed_rng)
    initial=[r for r in panel['windows'] if r['role']=='selection_initial']
    stream=BalancedStream(initial,config['seed'])
    reporter=None
    if rank==0:
        import wandb
        reporter=wandb.init(entity=config['wandb']['entity'],project=config['wandb']['project'],
            group=config['wandb']['group'],name=cfg['wandb_name'],id=ident[:12],resume='allow',config=config)
        (root/'control/wandb_url.txt').write_text(reporter.url+'\n')
    for sig in (signal.SIGTERM,signal.SIGUSR1):signal.signal(sig,request_stop)
    if start==0:
        baseline=evaluate(config,panel,model,branch,controller,reader,0,rank,world,full=False)
        native=evaluate_native(config,panel,model,branch,controller,0,rank,world)
        previous=baseline
        if rank==0:
            immutable(root/'runs/monitor_000000.json',baseline);immutable(root/'runs/native_000000.json',native)
    converged=False;stop=torch.tensor(0,device='cuda')
    for update in range(start,cfg['max_updates']):
        begin=time.monotonic();optimizer.zero_grad(set_to_none=True);metrics=None
        for micro in range(accumulation):
            position=update*batch+micro*world+rank
            row=stream.at(position);stratum=noise_stratum(config['seed'],position,len(panel['tasks']))
            parts=prepared(model,collate(reader,[row],[position]),config['seed'],[stratum])
            loss,terms=step_loss(model,branch,controller,parts,[row],[stratum],thresholds,update,config)
            (loss/accumulation).backward()
            metrics=terms/accumulation if metrics is None else metrics+terms/accumulation
            del loss,terms,parts
        norm=synchronize_controller(controller,cfg['gradient_clip']);optimizer.step();check_frozen(model,versions)
        dist.all_reduce(metrics);metrics/=world
        torch.cuda.synchronize();elapsed=time.monotonic()-begin;current=update+1
        if rank==0:
            names=['gt_action','teacher_action_applied','teacher_action_raw','kv_applied','kv_raw','mask_regularizer',
                'teacher_acceptance','teacher_worse_than_student','teacher_worse_than_zero','budget','ffn_floor','entropy','discreteness']
            log={'train/'+k:float(v) for k,v in zip(names,metrics[:13])}
            for layer in range(40):
                for j,kind in enumerate(('k','v')):log[f'kv/layer_{layer:02d}_{kind}']=float(metrics[13+layer*2+j])
            log.update(update=current,seconds_per_update=elapsed,mask_grad_norm=float(norm),
                unique_panel_windows=len(initial),mask_schedule_complete=current>=end,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            reporter.log(log,step=current)
            with (root/'runs/training.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
            print(json.dumps({k:log[k] for k in ('update','train/gt_action','train/kv_applied','seconds_per_update')}),flush=True)
        stop=torch.tensor(int(STOP or scheduler_stop_requested(root)),device='cuda');dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        if current%cfg['monitor_every']==0 or current==cfg['max_updates'] or bool(stop):
            monitor=evaluate(config,panel,model,branch,controller,reader,current,rank,world,full=False)
            if previous is not None:
                comparison=compare(previous,monitor,config);history.append(dict(update=current,**comparison));monitor['comparison']=comparison
            previous=monitor
            stable=stable_history(history,current,config)
            full=None;native=None;confirmation=None
            milestone=current>=end and (current%cfg['full_monitor_every']==0 or current==cfg['max_updates'])
            if stable or milestone:
                full=evaluate(config,panel,model,branch,controller,reader,current,rank,world,full=True)
                confirmation=confirm_full(previous_full,full,config);previous_full=full
                native=evaluate_native(config,panel,model,branch,controller,current,rank,world)
                converged=stable and confirmation['passed'] and native['quality']['no_resolved_regressions']
                monitor['full_confirmation']=confirmation;monitor['native_quality']=native['quality']
            # Read-only diagnostics never contribute optimizer gradients or advance the sampler.
            check_frozen(model,versions)
            rngs=gather(rng_state())
            state=dict(identity=ident,update=current,controller=controller.state_dict(),optimizer=optimizer.state_dict(),
                selection_config=dict(mask=config['mask'],seed=config['seed']),converged=converged,
                world_size=world,history=history,previous=previous,previous_full=previous_full,
                best_flow=best_flow,best_native=best_native,rng=rngs,next_global_sample=current*batch,
                thresholds_sha256=file_hash(Path(config['teacher_threshold_file'])))
            if rank==0:
                monitor_path=root/f'runs/monitor_{current:06d}.json'
                immutable(monitor_path,monitor)
                for kind,value in (('full_monitor',full),('native',native)):
                    if value is not None:immutable(root/f'runs/{kind}_{current:06d}.json',value)
                # Best checkpoints are immutable snapshots; latest remains the resumable state.
                checkpoint=root/f'runs/masks_{current:06d}.pt';save(checkpoint,state)
                checksum=file_hash(checkpoint)
                for kind,value in (('flow',full),('native',native)):
                    if value is None:continue
                    old=best_flow if kind=='flow' else best_native
                    if better(value['quality'],None if old is None else old['quality']):
                        item=dict(update=current,checkpoint=str(checkpoint),checkpoint_sha256=checksum,quality=value['quality'])
                        if kind=='flow':best_flow=item
                        else:best_native=item
                state['best_flow']=best_flow;state['best_native']=best_native;save(latest,state)
                report={}
                for suite,s in monitor['by_suite'].items():
                    for key in ('hard_error','soft_error','teacher_error','worst_task_ratio','retained_hidden_sensitivity','half_hidden_overlap','soft_to_binary_gap'):
                        report[f'monitor/{suite}/{key}']=s[key]
                    report[f'monitor/{suite}/active_sensitivity_groups']=sum(g['active'] for g in s['sensitivity_groups'])
                    for i,value in enumerate(s['retained_ffn_sensitivity']):report[f'monitor/{suite}/retained_ffn_{i:02d}']=value
                for task,t in monitor['by_task'].items():
                    for k,v in t.items():report[f'task/{task}/{k}']=v
                if native:
                    report['native/mean_task_ratio']=native['quality']['mean_task_ratio']
                    report['native/resolved_regressions']=len(native['quality']['resolved_regressions'])
                    for task,t in native['quality']['by_task'].items():report[f'native/{task}/ratio']=t['ratio_to_teacher']
                reporter.log(report,step=current)
            metadata=[best_flow,best_native] if rank==0 else [None,None]
            dist.broadcast_object_list(metadata,src=0);best_flow,best_native=metadata
            barrier()
        if converged or bool(stop):
            if rank==0:immutable(root/f'control/stop_{current:06d}.json',dict(update=current,converged=converged,
                reason='stability_and_action_quality' if converged else 'scheduler_signal',policy_retention_established=False))
            break
    if rank==0:
        if not bool(stop):
            finalize_selection(root,ident,best_native,best_flow,current,converged,cfg['max_updates'])
        if reporter:reporter.finish()
    barrier()

if __name__=='__main__':main()
