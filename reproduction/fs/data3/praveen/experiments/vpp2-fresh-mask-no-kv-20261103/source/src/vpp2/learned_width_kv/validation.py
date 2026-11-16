"""Fixed paired offline validation, separate soft-mask and exact-budget models."""
import json
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.utils.data import default_collate
from vpp2.compact_kv.model import prepare_batch,native_cache,action_prediction,infer_from_cache
from vpp2.compact_kv.losses import action_error,kv_terms,kv_plane_terms
from vpp2.compact_kv.spec import atomic_json
from vpp2.compact_kv.checkpointing import rng_state,restore_rng
from .runtime import student_cache
from .reliability import action_losses,kv_losses
from .data_scope import aggregate_validation, task_scope


def cpu(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu()
    if isinstance(value,dict): return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu(v) for v in value]
    return value


def action_metrics(student_action, teacher_action, target, valid, time_weight, task, strata, thresholds, config, kv_errors=None):
    training, excluded, _ = task_scope(config)
    if set(thresholds) != set(training):
        raise RuntimeError('Validation thresholds must match the training tasks exactly')
    result = dict(
        student_flow_gt=float((action_error(student_action, target, valid) * time_weight).mean()),
        teacher_flow_gt=float((action_error(teacher_action, target, valid) * time_weight).mean()),
        action_teacher=float((action_error(student_action, teacher_action, valid) * time_weight).mean()),
        action_teacher_weighted=None, teacher_action_weight=None, teacher_reliability=None,
        fixed_teacher_group='excluded_uncalibrated', teacher_kv_weight=None,
        teacher_supervision_rejected_fraction=None, kv_gated=None, kv_gated_by_layer_kv=None)
    if task in training:
        _, kd, _, weight, reasons = action_losses(student_action, teacher_action, target, valid,
                                                  time_weight, [task], strata, thresholds)
        result.update(action_teacher_weighted=float(kd), teacher_action_weight=float(weight.mean()),
                      teacher_reliability=float(reasons['reliability'].mean()), teacher_kv_weight=float(weight.mean()),
                      teacher_supervision_rejected_fraction=float((weight==0).float().mean()))
        if kv_errors is not None:
            _, applied, layers = kv_losses(kv_errors, weight)
            result.update(kv_gated=float(applied), kv_gated_by_layer_kv=layers.cpu().tolist())
        error = float(action_error(teacher_action, target, valid))
        q = thresholds[task][strata[0]]
        result['fixed_teacher_group'] = 'reliable' if error <= q['q75'] else ('unreliable' if error >= q['q95'] else 'intermediate')
    elif task not in excluded:
        raise RuntimeError('Unknown validation task: ' + task)
    return result


def chunk_metrics(student, teacher, target, is_pad):
    """Keep the primary 32-step metric; expose executed-prefix and dimension errors."""
    if student.shape != teacher.shape or student.shape != target.shape or tuple(target.shape[1:]) != (32,16):
        raise ValueError('Expected native RoboDojo action chunks [B,32,16]')
    valid = torch.ones(target.shape[:2],device=target.device,dtype=torch.bool) if is_pad is None else ~is_pad.bool()
    if valid.shape != target.shape[:2] or not (valid.sum(-1)>0).all():
        raise ValueError('Action chunk metrics require real demonstrated action steps')
    result = {}
    for name, prediction, truth in [('student',student,target),('teacher',teacher,target),('chunk_teacher',student,teacher)]:
        difference = (prediction.float()-truth.float()).square()
        per_dimension = ((difference*valid[...,None]).sum(1)/valid.sum(1,keepdim=True)).mean(0)
        if not torch.isfinite(per_dimension).all():
            raise FloatingPointError('Nonfinite generated action chunk')
        result[name+'_mse_by_action_dimension'] = per_dimension.cpu().tolist()
        prefix_mask = ~valid[:,:24]
        result[name+'_executed_prefix24_mse'] = float(action_error(prediction[:,:24],truth[:,:24],prefix_mask).mean())
    return result


@torch.no_grad()
def evaluate(engine,controller,reference,panel,config,phase,update,directory,thresholds,full=False,mode='soft',limit=False):
    saved_rng=rng_state()
    rank,world=dist.get_rank(),dist.get_world_size(); start=time.monotonic()
    items=list(range(128)) if not limit else [i*16 for i in range(8)]
    engine.eval();reference.mot.training=False
    local=[]
    for ordinal,item_id in enumerate(items[rank::world]):
        index,task_id,trajectory,step=panel.items[item_id];task=panel.tasks[task_id]
        sample=panel.dataset._get(index)
        if (int(sample['trajectory_idx']),int(sample['step_idx']))!=(trajectory,step):
            raise RuntimeError('Validation sample identity changed')
        position=-(item_id+1);sample['pilot_position']=torch.tensor(position)
        inputs,noise,tv,noisy,ta,target,mask,u=prepare_batch(reference,default_collate([sample]),config['seed'],return_noise_u=True)
        teacher=native_cache(reference,inputs,noise,tv,mask)
        student,_=student_cache(engine,controller,phase,update,inputs,noise,tv,mask,mode)
        teacher_action=action_prediction(reference,inputs,teacher,noisy,ta,mask)
        student_action=action_prediction(reference,inputs,student,noisy,ta,mask)
        valid=inputs.get('action_loss_is_pad',inputs.get('action_is_pad'))
        time_weight=reference.train_action_scheduler.training_weight(ta)
        strata=(u*4).long().clamp_max(3).tolist()
        terms=kv_terms(student,teacher)
        metrics=action_metrics(student_action,teacher_action,target,valid,time_weight,task,strata,thresholds,config,terms)
        row=dict(item_id=item_id,task=task,trajectory=trajectory,step=step,**metrics,kv_raw=float(terms.mean()),
            kv_by_layer_kv=terms.mean(0).cpu().tolist(),kv_by_plane_layer_kv=kv_plane_terms(student,teacher).mean(0).cpu().tolist())
        if full:
            tc=infer_from_cache(reference,inputs,teacher,mask,[position],config['seed'])
            sc=infer_from_cache(reference,inputs,student,mask,[position],config['seed'])
            real=inputs.get('action_is_pad')
            row.update(teacher_chunk_mse=float(action_error(tc,inputs['action'],real)),
                       student_chunk_mse=float(action_error(sc,inputs['action'],real)),
                       chunk_teacher_mse=float(action_error(sc,tc,real)))
            row.update(chunk_metrics(sc,tc,inputs['action'],real))
            if ordinal==0:
                folder=Path(directory)/'validation';folder.mkdir(parents=True,exist_ok=True)
                fixture=dict(inputs=cpu(inputs),noise=cpu(noise),t_v=cpu(tv),noisy_action=cpu(noisy),t_a=cpu(ta),
                    mask=cpu(mask),position=position,expected_flow=cpu(student_action),expected_chunk=cpu(sc),update=update,mode=mode)
                if phase=='A' and mode=='binary' and update==1000:
                    fixture['expected_cache']=cpu(student)
                torch.save(fixture,folder/f'fixture_{update:06d}_{mode}_rank{rank}.pt')
        local.append(row)
        del student,teacher,inputs
    gathered=[None]*world;dist.all_gather_object(gathered,local)
    result=None
    if rank==0:
        rows=sorted([r for part in gathered for r in part],key=lambda r:r['item_id'])
        if len({r['item_id'] for r in rows})!=len(items): raise RuntimeError('Duplicate/missing validation windows')
        summary=aggregate_validation(rows,config)
        result=dict(kind='validation_'+mode,phase=phase,phase_update=update,update=update+(1000 if phase=='B' else 0),
            windows=len(rows),seconds=time.monotonic()-start,full=full,**summary)
        calibrated=[r for r in rows if r['fixed_teacher_group']!='excluded_uncalibrated']
        keys=[k for k,v in calibrated[0].items() if isinstance(v,float)]
        def average(group): return {k:sum(r[k] for r in group)/len(group) for k in keys}
        result['calibrated_windows']=len(calibrated)
        result['uncalibrated_diagnostic_windows']=len(rows)-len(calibrated)
        result['by_teacher_group']={group:average([r for r in calibrated if r['fixed_teacher_group']==group])
            for group in ('reliable','intermediate','unreliable') if any(r['fixed_teacher_group']==group for r in calibrated)}
        train,excluded,_=task_scope(config)
        for name,tasks in [('primary',train),('excluded_diagnostic',excluded),('full_precision',train+excluded)]:
            scoped=[r for r in rows if r['task'] in tasks]
            if scoped:
                result['by_scope'][name]['kv_by_layer_kv']=torch.tensor([r['kv_by_layer_kv'] for r in scoped]).mean(0).tolist()
                # Excluded tasks have no calibrated gate: do not report zeros or
                # an accepted-only mean as an all-task gated metric.
                applied=[r['kv_gated_by_layer_kv'] for r in scoped]
                result['by_scope'][name]['kv_gated_by_layer_kv']=(torch.tensor(applied).mean(0).tolist()
                    if all(value is not None for value in applied) else None)
        result['kv_by_layer_kv']=torch.tensor([r['kv_by_layer_kv'] for r in rows]).mean(0).tolist()
        result['kv_gated_by_layer_kv']=result['by_scope']['full_precision']['kv_gated_by_layer_kv']
        result['kv_by_plane_layer_kv']=torch.tensor([r['kv_by_plane_layer_kv'] for r in rows]).mean(0).tolist()
        atomic_json(Path(directory)/f'validation/update_{update:06d}_{mode}.json',dict(summary=result,windows=rows))
    dist.barrier();engine.train();reference.mot.training=True
    restore_rng(saved_rng)
    return result
