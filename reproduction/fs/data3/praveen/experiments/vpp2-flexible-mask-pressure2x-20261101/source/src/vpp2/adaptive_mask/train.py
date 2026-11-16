"""One frozen-policy mask trajectory, bounded rescue, and explicit quality receipts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import signal
import time
import os
import torch
import torch.distributed as dist
from vpp2.full_mask.panel import BalancedStream,noise_stratum,immutable
from .controller import synchronize_controller
from .runtime import distributed,initialize,barrier,gather,step,check_frozen,model_versions
from .state import identity,rng_state,restore_rng,snapshot,restore,save,diagnostic_rng
from .data import prepare
from .budget import Budget
from .priority import MaskPriority
from .hidden import HiddenSearch
from .feedback import FittingFeedback,selection_churn
from .quality import stable
from .quality_policy import policy_mode, LEGACY, ACTION_ONLY
from .monitor import evaluate,native_evaluate,gradient_diagnostics

STOP=False

def stop_request(*args):
    global STOP
    STOP=True

def reduce_metrics(values):
    keys=sorted(values)
    values=torch.stack([torch.as_tensor(values[k],device='cuda',dtype=torch.float64) for k in keys])
    if dist.is_initialized():dist.all_reduce(values);values/=dist.get_world_size()
    return dict(zip(keys,values.cpu().tolist()))

def require_receipts(config,ident,world):
    for name in ('cpu','native','distributed'):
        rec=json.loads((Path(config['root'])/'checks'/f'{name}.json').read_text())
        if not rec.get('passed') or rec.get('identity')!=ident:raise ValueError('Missing/stale '+name+' check')
    if world!=8 or config['train']['global_batch']!=256 or config['train']['micro_batch']!=1:
        raise ValueError('Unverified execution geometry')

def combined_quality(monitor,native):
    return {k:(any if k=='resolved_violation' else all)(x[k] for x in (monitor,native))
            for k in ('passed','clearly_passing','resolved_violation')}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    config=json.loads(Path(args.config).read_text());root=Path(config['root']);panel=json.loads(Path(config['panel_file']).read_text())
    from .preflight import validate_inputs
    validate_inputs(config,panel)
    rank,world=distributed();ident=identity(config,panel);require_receipts(config,ident,world)
    from vpp2.critical_windows.inference import model_identity
    hashes=[model_identity(config) if rank==0 else None]
    if dist.is_initialized():dist.broadcast_object_list(hashes,src=0)
    if hashes[0]!=json.loads((root/'checks/native.json').read_text())['checkpoint_hashes']:
        raise ValueError('Native checkpoint changed since preflight')
    model,reader,controller,branch,inventory=initialize(config);versions=model_versions(model)
    cfg=config['train'];bc=config['budget']
    optimizer=torch.optim.AdamW(controller.trainable_parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['betas']),weight_decay=0.)
    priority=MaskPriority(controller,**config['mask_priority'])
    hidden=HiddenSearch(**config['hidden_search'])
    budget=Budget(original=inventory.original,target=bc['target_parameters'],information_updates=cfg['information_updates'],
        milestones=bc['milestones'],stability_updates=bc['stability_updates'],extra_updates=bc['extra_updates'],early_full_update=bc['early_full_update'],feedback_every=config['capacity_feedback']['every'])
    if budget.maximum_updates!=cfg['max_attempted_updates']:raise ValueError('Resource bound differs from schedule')
    initial=[r for r in panel['windows'] if r['role']=='selection_initial'];stream=BalancedStream(initial,config['seed'])
    cursor=lineage=pruning_updates=0;coverage=Counter();attempt_coverage=Counter();history=[]
    feedback=FittingFeedback(initial);feedback_selection=controller.selection()
    parent=None;parent_full=None;parent_native=None;parent_priority=None;full_checks=0
    latest=root/'runs/latest.pt';finished=terminated=False;quality=None;pending_check=False
    tracking_run_id=ident[:12];execution_revisions=[];protocol_revisions=[]
    quality_gate_mode=policy_mode(config)
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        if saved['identity']!=ident or saved['world']!=world:raise ValueError('Resume provenance mismatch')
        tracking_run_id=saved.get('tracking_run_id',ident[:12]);execution_revisions=saved.get('execution_revisions',[])
        protocol_revisions=saved.get('protocol_revisions',[])
        if saved.get('quality_gate_mode',LEGACY)!=quality_gate_mode:
            raise ValueError('Quality policy changed without an explicit checkpoint migration')
        cursor,lineage,coverage,pruning_updates=restore(saved['state'],controller,optimizer);restore_rng(saved['rngs'][rank])
        budget=Budget(**saved['budget']);hidden=HiddenSearch(**saved['hidden_search']);attempt_coverage=Counter(saved['attempt_coverage']);history=saved['history']
        parent=saved['parent']
        if parent is not None:parent['rng']=saved['parent_rngs'][rank]
        parent_full=saved['parent_full'];parent_native=saved['parent_native'];full_checks=saved['full_checks']
        priority.load_state_dict(saved['priority']);parent_priority=saved['parent_priority']
        feedback=FittingFeedback(initial,saved['fitting_feedback']);feedback_selection=saved['feedback_selection']
        pending_check=saved.get('pending_check',False)
        if saved.get('finished') or saved.get('terminated'):return
    reporter=None
    if rank==0:
        import wandb
        reporter=wandb.init(entity=config['wandb']['entity'],project=config['wandb']['project'],group=config['wandb']['group'],
            name=config['wandb']['name'],id=tracking_run_id,resume='allow',config=config)
        reporter.define_metric('attempt');reporter.define_metric('*',step_metric='attempt')
        (root/'control').mkdir(parents=True,exist_ok=True);(root/'control/wandb_url.txt').write_text(reporter.url+'\n')
        print(json.dumps(dict(event='quality_gate_protocol',attempt=budget.attempt,lineage_update=lineage,
            quality_gate_mode=quality_gate_mode,protocol_revisions=protocol_revisions)),flush=True)
    for sig in (signal.SIGUSR1,signal.SIGTERM):signal.signal(sig,stop_request)
    if budget.attempt==0:
        with diagnostic_rng():
            parent_full=evaluate(config,panel,model,reader,branch,controller,0,rank,world,full=True)
            parent_native=native_evaluate(config,panel,model,branch,controller,rank,world)
        if not combined_quality(parent_full,parent_native)['passed']:raise RuntimeError('Unpruned baseline fails numerical quality check')
        if rank==0:immutable(root/'runs/baseline.json',dict(monitor=parent_full,native=parent_native))
    stop=torch.zeros((),device='cuda',dtype=torch.int32)
    while (budget.attempt<cfg['max_attempted_updates'] or pending_check) and not (finished or terminated):
        if not pending_check:
            projected=hidden.project(controller,inventory,budget.cap);priority.refresh()
            optimizer.zero_grad(set_to_none=True)
            lr=cfg['learning_rate']*min((lineage+1)/cfg['warmup_updates'],1.)
            for group in optimizer.param_groups:group['lr']=lr
            begin=time.monotonic();metrics={};local_feedback=[];local_values=[]
            accumulation=cfg['global_batch']//world
            active_pruning=projected['parameters']<inventory.original
            for micro in range(accumulation):
                position=cursor+micro*world+rank;row=stream.at(position)
                stratum=noise_stratum(config['seed'],position,len(panel['tasks']))
                parts=prepare(model,reader,[row],[position],config['seed'],[stratum])
                loss,terms,details,_=step(model,branch,controller,parts,[row],pruning_updates+int(active_pruning),config)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training objective')
                (loss/accumulation).backward()
                for key,value in terms.items():metrics[key]=metrics.get(key,0)+torch.as_tensor(value,device='cuda').detach()/accumulation
                local_feedback.append(dict(window_id=row['id'],position=position,stratum=int(stratum)))
                local_values.append(torch.stack([details['teacher_error'][0].detach(),details['student_error'][0].detach()]))
                del loss,terms,details,parts
            norm=synchronize_controller(controller,cfg['gradient_clip'])
            priority.step(optimizer);check_frozen(model,versions);optimizer.zero_grad(set_to_none=True)
            cursor+=cfg['global_batch'];lineage+=1;pruning_updates+=int(active_pruning);budget.advance(hold_capacity=hidden.active)
            for row,values in zip(local_feedback,torch.stack(local_values).cpu().tolist()):
                row.update(teacher_error=values[0],student_error=values[1])
            global_feedback=sum(gather(local_feedback),[]);feedback.add(global_feedback)
            ids=[r['window_id'] for r in global_feedback];coverage.update(ids);attempt_coverage.update(ids)
            del local_values,global_feedback
            projected=hidden.project(controller,inventory,budget.cap)
            priority.refresh()
            log=reduce_metrics(metrics);log.update(priority.summary());torch.cuda.synchronize()
            log.update(attempt=budget.attempt,lineage_update=lineage,schedule_update=budget.progress,phase=budget.phase,
                cap=budget.cap,parameters=projected['parameters'],hidden_width=hidden.width,hidden_trial_active=hidden.active,learning_rate=lr,mask_gradient_norm=float(norm),
                unique_windows_lineage=len(coverage),unique_windows_attempted=len(attempt_coverage),
                windows_seen_twice_lineage=sum(v>=2 for v in coverage.values()),seconds_per_update=time.monotonic()-begin,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                **{'protocol/action_only_gate':int(quality_gate_mode==ACTION_ONLY),'protocol/revision_count':len(protocol_revisions)})
            if rank==0:
                reporter.log(log)
                with (root/'runs/training.jsonl').open('a') as f:f.write(json.dumps(log,allow_nan=False)+'\n')
                print(json.dumps({k:log[k] for k in ('attempt','lineage_update','parameters','loss/gt_action','loss/late_kv_applied','seconds_per_update')}),flush=True)
            # The all-retained collection phase learns only rankings, without changing predictions.
            if budget.progress==cfg['information_updates'] and parent is None:
                parent=snapshot(controller,optimizer,cursor,lineage,coverage,pruning_updates);parent_priority=priority.state_dict()
        stop.fill_(int(STOP));request=root/'control/stop_request.json'
        if request.exists() and str(json.loads(request.read_text()).get('job_id'))==os.environ.get('SLURM_JOB_ID'):stop.fill_(1)
        if dist.is_initialized():dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        check=budget.attempt%cfg['monitor_every']==0 or budget.attempt==cfg['max_attempted_updates']
        if check and not bool(stop):
            fit_report=feedback.report(config['quality'],config['capacity_feedback']['minimum_episodes_per_task_stratum'])
            churn=selection_churn(feedback_selection,controller.selection())
            do_full=budget.full_due() or hidden.full_due(budget.attempt)
            if do_full:
                full_checks+=1
                if full_checks>cfg['max_full_checks_excluding_initial']:raise RuntimeError('Unexpected expensive-check count')
            with diagnostic_rng():
                monitor=evaluate(config,panel,model,reader,branch,controller,pruning_updates,rank,world,full=do_full,
                    parent=parent_full if do_full or hidden.active else None)
                diagnostics=gradient_diagnostics(config,panel,model,reader,branch,controller,pruning_updates,rank,world,full=do_full,
                    check_index=budget.attempt//cfg['monitor_every']-1)
                native=native_evaluate(config,panel,model,branch,controller,rank,world,parent=parent_native) if do_full else None
            quality=combined_quality(monitor,native) if native else monitor
            priority.observe(pruning_active=inventory.count(controller.selection())<inventory.original)
            hidden.observe(controller)
            if fit_report['resolved_violation']:priority.reopen()
            record=dict(lineage_update=lineage,attempt=budget.attempt,cap=budget.cap,selection=controller.selection(),
                full_quality_checked=do_full,quality_gate_mode=quality_gate_mode,quality_passed=quality['passed'],screen_clear=not quality['resolved_violation'] and not fit_report['resolved_violation'],
                gradients_active=all(all(f['active'] for f in d['families'].values()) for d in diagnostics))
            history.append(record);decision='observe';restored=False
            trial_decision=hidden.assess(budget.attempt,quality,fit_report,do_full,churn) if hidden.active else None
            if trial_decision and trial_decision.startswith('reject_hidden'):
                if parent is None:raise AssertionError('Hidden trial has no accepted parent')
                cursor,lineage,coverage,pruning_updates=restore(parent,controller,optimizer)
                priority.load_state_dict(parent_priority);priority.reopen();history=[]
                hidden.restored(controller.selection(),trial_decision)
                budget.cap=budget.accepted_cap;budget.authorized_updates=0
                # Restored parent is fully checked; rejected trial cannot update it.
                budget.selection_hold=False;budget.rescue_remaining=0
                decision=trial_decision;restored=True
            elif hidden.active and trial_decision!='accept_hidden_width':
                decision=trial_decision
            elif do_full:
                if trial_decision=='accept_hidden_width':hidden.accept()
                covered=len(coverage)==len(initial) and min(coverage.values())>=2
                decision=budget.decide(quality['passed'],stable(history),covered,resolved_violation=quality['resolved_violation'],
                    extra_quality_check=not budget.full_due())
                if quality['passed']:
                    parent=snapshot(controller,optimizer,cursor,lineage,coverage,pruning_updates);parent_full=monitor;parent_native=native;parent_priority=priority.state_dict()
                if decision=='restore_parent_and_stop_shrinking':
                    if parent is None:raise AssertionError('No validated parent')
                    cursor,lineage,coverage,pruning_updates=restore(parent,controller,optimizer);history=[]
                    priority.load_state_dict(parent_priority);priority.reopen()
                    hidden.restored(controller.selection(),'restore_accepted_parent');restored=True
                elif decision in ('rescue_confirmed_harm_once','extend_final_check_once'):
                    history=[]
                elif decision=='finish':finished=True
                elif decision in ('stop_not_stable','stop_quality_failed','stop_quality_unresolved'):terminated=True
                if trial_decision=='accept_hidden_width':decision='accept_hidden_width'
                if not (finished or terminated or restored) and quality['passed']:
                    if hidden.start(budget.attempt,controller,inventory,budget,fit_report,churn,quality_passed=record['gradients_active']):
                        history=[];priority.reopen();decision='begin_isolated_hidden_trial'
            feedback_decision=budget.observe_fitting(fit_report,churn,
                config['capacity_feedback']['max_ffn_churn'],config['capacity_feedback']['max_cross_head_churn'],
                config['capacity_feedback']['max_hidden_churn'],hold_capacity=hidden.active or restored)
            feedback=FittingFeedback(initial) if restored else feedback
            feedback.clear();feedback_selection=controller.selection()
            check_frozen(model,versions)
            if rank==0:
                immutable(root/f'runs/monitor_attempt_{budget.attempt:06d}.json',dict(monitor=monitor,native=native,diagnostics=diagnostics,
                    record=record,decision=decision,budget=budget.state_dict(),priority=priority.summary(),
                    fitting_feedback=fit_report,channel_churn=churn,feedback_decision=feedback_decision,
                    hidden_search=hidden.state_dict(),next_selection=controller.selection()))
                from .report import log_monitor,log_feedback
                log_feedback(reporter,fit_report,churn,budget,decision,budget.attempt,root)
                log_monitor(reporter,monitor,native,diagnostics,record['selection'],budget.attempt,root)
                reporter.log({'attempt':budget.attempt,'hidden/width':hidden.width,'hidden/trial_active':int(hidden.active),
                    'hidden/trials':hidden.trials,'hidden/blocked':int(hidden.blocked),'hidden/turnover':churn['hidden'],
                    'hidden/decision':hidden.last_decision})
        pending_check=bool(check and stop)
        # A boundary stop saves its pending check; resume performs it before another update.
        if check or bool(stop):
            local=snapshot(controller,optimizer,cursor,lineage,coverage,pruning_updates)
            rngs=gather(rng_state());parent_rngs=gather(None if parent is None else parent['rng'])
            saved=dict(identity=ident,tracking_run_id=tracking_run_id,execution_revisions=execution_revisions,
                protocol_revisions=protocol_revisions,quality_gate_mode=quality_gate_mode,world=world,state=local,rngs=rngs,budget=budget.state_dict(),hidden_search=hidden.state_dict(),attempt_coverage=dict(attempt_coverage),
                history=history,parent=parent,parent_rngs=parent_rngs,parent_full=parent_full,parent_native=parent_native,
                full_checks=full_checks,finished=finished,terminated=terminated,priority=priority.state_dict(),parent_priority=parent_priority,
                fitting_feedback=feedback.state_dict(),feedback_selection=feedback_selection,pending_check=pending_check)
            if rank==0:
                save(root/f'runs/masks_attempt_{budget.attempt:06d}.pt',saved);save(latest,saved)
                if finished:
                    if hidden.active or len(controller.selection()['hidden_indices'])!=hidden.width:raise AssertionError('Unresolved hidden trial at finalization')
                    immutable(root/'final_selection.json',dict(selection=controller.selection(),parameters=inventory.count(controller.selection()),
                        identity=ident,attempt=budget.attempt,lineage_update=lineage,quality=quality,stable=True,
                        quality_gate_mode=quality_gate_mode,protocol_revisions=protocol_revisions,
                        hidden_search=hidden.state_dict(),
                        compression_achieved=inventory.count(controller.selection())<inventory.original,policy_retention_established=False,
                        audit_manifest_sha256=config['artifacts']['audit_manifest']['sha256'],
                        monitoring_role='development_selection_not_independent_audit'))
            barrier()
        if bool(stop):break
    if rank==0:
        immutable(root/f'control/status_attempt_{budget.attempt:06d}_{"finished" if finished else "interrupted" if bool(stop) else "not_finalized"}.json',dict(finished=finished,attempt=budget.attempt,lineage_update=lineage,
            reason='stability_and_quality' if finished else 'interrupted' if bool(stop) else 'resource_or_stability_limit_not_finalized',
            calibration_passes=0,full_checks_excluding_initial=full_checks,simulation_started=False,compact_recovery_started=False))
        reporter.finish()
    barrier()

if __name__=='__main__':main()
