"""Read-only per-objective weight gradients, reduced over disjoint ZeRO shards.

No optimizer steps. Diagnostic RNG and engine counters are restored. Model and
Adam master/moment tensors are fingerprinted before/after; a GPU toy verifies
that a subsequent normal optimizer update is unchanged by these diagnostics.
"""
from collections import defaultdict
import copy
import math
from pathlib import Path
import re
import time

import torch
import torch.distributed as dist

from .rng import diagnostic_rng


def clear_gradients(engine):
    opt=engine.optimizer
    opt.zero_grad()
    # With ZeRO-3, zero_grad clears Parameter.grad but leaves gradient partitions.
    # Explicitly clear these so a term that has a zero derivative cannot inherit
    # another term's old gradient. No optimizer state/parameter is changed here.
    for p in engine.module.parameters():
        if p.requires_grad:
            opt.get_local_fp32_grad_for_param(p).zero_()
    opt.averaged_gradients.clear()


def layer_name(name):
    match=re.search(r'(?:blocks|layers)\.(\d+)\.',name)
    return 'block_'+match.group(1).zfill(2) if match else 'other'


def capture_local(engine):
    out={}
    for name,p in engine.module.named_parameters():
        if p.requires_grad:
            v=engine.optimizer.get_local_fp32_grad_for_param(p).detach().float().cpu().clone()
            if not torch.isfinite(v).all():raise FloatingPointError('Nonfinite gradient: '+name)
            out[name]=v
    return out


def summarize(gradients, device):
    names=('gt','video','kv');groups=defaultdict(lambda:torch.zeros(6,dtype=torch.float64))
    for parameter in gradients['gt']:
        g=[gradients[k][parameter].double().reshape(-1) for k in names]
        values=torch.stack([torch.dot(g[0],g[0]),torch.dot(g[1],g[1]),torch.dot(g[2],g[2]),
                            torch.dot(g[0],g[1]),torch.dot(g[0],g[2]),torch.dot(g[1],g[2])])
        groups[layer_name(parameter)]+=values;groups['all']+=values
    keys=sorted(groups);a=torch.stack([groups[k] for k in keys]).to(device)
    # Gradients have already been averaged over data-parallel ranks. Each rank
    # now owns a distinct coordinate shard: SUM squared norms/dots, not averages.
    dist.all_reduce(a,op=dist.ReduceOp.SUM)
    result={}
    for key,v in zip(keys,a.cpu().tolist()):
        norms=[math.sqrt(max(x,0)) for x in v[:3]]
        result[key]={k+'_norm':n for k,n in zip(names,norms)}
        for name,i,j,dot in [('gt_video',0,1,v[3]),('gt_kv',0,2,v[4]),('video_kv',1,2,v[5])]:
            result[key][name+'_cosine']=None if norms[i]*norms[j]<1e-20 else max(-1.,min(1.,dot/(norms[i]*norms[j])))
    return result


def fitting_rows(panel):
    from .data import validate_panel
    from vpp2.full_mask.panel import seed_for
    fit,_=validate_panel(panel);by=defaultdict(list)
    for r in fit:by[r['task']].append(r)
    chosen=[]
    for task in sorted(by):
        ranked=sorted(by[task],key=lambda r:seed_for(100008,'low_lr_gradient_panel',r['id']))
        first=ranked[0];second=next(r for r in ranked[1:] if r['episode']!=first['episode'])
        chosen.extend([first,second])
    if len(chosen)!=70:raise ValueError('Gradient panel must include two fitting windows per task')
    return chosen


def engine_counters(engine):
    keys=('micro_steps','global_steps','global_samples','skipped_steps','_is_gradient_accumulation_boundary',
          '_scale_wrt_gas','_global_grad_norm','losses','total_loss','mean_loss')
    out={}
    for k in keys:
        if hasattr(engine,k):
            v=getattr(engine,k);out[k]=v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v)
    return out


def restore_counters(engine,state):
    for k,v in state.items():setattr(engine,k,v)
    engine.optimizer.is_gradient_accumulation_boundary=engine.is_gradient_accumulation_boundary()


def measured_gradients(engine,make_terms,batches,world,valid_count,device,active_names=('gt','video','kv')):
    """batches: iterable input descriptors; make_terms returns three weighted losses.

    Losses are per one-window batch; multiply by world/valid_count because
    DeepSpeed averages over ranks. Padded windows must return zero multiplier.
    """
    counters=engine_counters(engine);saved={}
    try:
        with diagnostic_rng():
            for name in active_names:
                clear_gradients(engine)
                for batch in batches:
                    engine.set_gradient_accumulation_boundary(False)
                    terms,valid=make_terms(batch)
                    # All graphs are traversed so ZeRO's checkpointed backward
                    # hooks and unused-parameter handling are identical per term.
                    value=terms[name]+sum(v*0. for k,v in terms.items() if k!=name)
                    engine.backward(value*(world/valid_count if valid else 0.),scale_wrt_gas=False)
                    del terms,value
                saved[name]=capture_local(engine)
            for name in ('gt','video','kv'):
                if name not in saved: saved[name]={k:torch.zeros_like(v) for k,v in saved['gt'].items()}
            result=summarize(saved,device)
    finally:
        saved.clear();clear_gradients(engine);restore_counters(engine,counters)
    return result


def adam_state(engine):
    # ZeRO-3 clears the underlying Adam param_groups between updates. Calling
    # Adam.state_dict() directly can fail after an update; index states by the
    # stable master-partition order without changing any optimizer structure.
    opt=engine.optimizer
    return dict(states=[opt.optimizer.state.get(p,{}) for p in opt.fp32_partitioned_groups_flat],
                groups=[{k:v for k,v in g.items() if k!='params'} for g in opt.optimizer.param_groups])


def immutable_fingerprint(runtime):
    from .preflight import fingerprint
    opt=runtime.engine.optimizer
    # Adam parameters are partitioned, so this does not gather a full model.
    return fingerprint(dict(master=opt.fp32_partitioned_groups_flat,
        moments=adam_state(runtime.engine),
        student={n:getattr(p,'ds_tensor',p) for n,p in runtime.engine.module.named_parameters()},
        reference={n:getattr(p,'ds_tensor',p) for n,p in runtime.reference.named_parameters()}))


def scalar_summary(rows):
    """Ranges over individual diagnostic windows, not a training trajectory."""
    if not rows:raise ValueError('No scalar diagnostic measurements')
    keys=('gt_action_loss','teacher_video_raw','teacher_video_gated',
          'weighted_gt_action_loss','weighted_teacher_video_loss','reliability')
    result={}
    for key in keys:
        values=torch.tensor([r[key] for r in rows],dtype=torch.float64)
        if not torch.isfinite(values).all():raise FloatingPointError('Invalid scalar diagnostic')
        result[key]=dict(min=float(values.min()),p50=float(values.quantile(.5)),
            p95=float(values.quantile(.95)),max=float(values.max()),mean=float(values.mean()))
    result['samples']=len(rows)
    return result


def run(runtime,panel,update):
    from .config import identity
    from .preflight import fingerprint
    from .rng import rng_state
    from vpp2.compact_kv.spec import atomic_json
    from vpp2.full_mask.panel import digest
    root=Path(runtime.config['root']);p=root/'recovery/diagnostics'/f'gradients_{update:06d}.json'
    rows=fitting_rows(panel);started=time.monotonic();before=immutable_fingerprint(runtime)
    random_before=fingerprint(rng_state());steps=runtime.engine.global_steps
    records=[];loss_records=[]
    for draw in range(2):
        batches=[]
        for base in range(0,len(rows),runtime.world):
            i=base+runtime.rank;valid=i<len(rows);row=rows[i if valid else 0]
            batches.append((row,valid,draw))
        seen=set();local_scalars=[]
        def terms(batch):
            row,valid,noise_draw=batch
            position=-9_000_000-noise_draw*1_000_000-row['id']
            _,metrics,details,values=runtime.forward([row],[position],update)
            w=runtime.config['train']
            if valid and row['id'] not in seen:
                seen.add(row['id'])
                vector=torch.stack([values['gt'].detach(),metrics['teacher_video_raw'].detach(),
                    values['video'].detach(),w['action_gt_weight']*values['gt'].detach(),
                    w['teacher_video_weight']*values['video'].detach(),details['reliability'].mean().detach()]).double()
                local_scalars.append((dict(window_id=row['id'],task=row['task'],suite=row['suite'],noise_draw=noise_draw),vector))
            return dict(gt=w['action_gt_weight']*values['gt'],
                        video=w['teacher_video_weight']*values['video'],
                        kv=w['kv_weight']*values['kv']),valid
        result=measured_gradients(runtime.engine,terms,batches,runtime.world,len(rows),runtime.device,
            active_names=('gt','video') if runtime.config['train']['kv_weight']==0. else ('gt','video','kv'))
        keys=('gt_action_loss','teacher_video_raw','teacher_video_gated','weighted_gt_action_loss','weighted_teacher_video_loss','reliability')
        own=[dict(meta,**dict(zip(keys,vector.cpu().tolist()))) for meta,vector in local_scalars]
        gathered=[None]*runtime.world;dist.all_gather_object(gathered,own)
        merged=sorted(sum(gathered,[]),key=lambda x:x['window_id'])
        if len(merged)!=70 or len({r['window_id'] for r in merged})!=70:
            raise ValueError('Missing or duplicated diagnostic loss windows')
        loss_records.extend(merged)
        records.append(dict(noise_draw=draw,groups=result,loss_summary=scalar_summary(merged)))
    unchanged=(before==immutable_fingerprint(runtime) and random_before==fingerprint(rng_state())
               and runtime.engine.global_steps==steps)
    if not unchanged:raise RuntimeError('Gradient diagnostics changed model/optimizer/RNG state')
    runtime.check_frozen()
    result=dict(identity=identity(runtime.config),update=update,passed=True,
        training_state_unchanged=True,windows=70,noise_draws=2,source='training fitting split only',
        window_identity=digest([(r['id'],r['episode'],r['frame']) for r in rows]),
        coordinate_scope='all trainable compact video parameters; disjoint ZeRO shards summed',
        weighting=dict(gt=runtime.config['train']['action_gt_weight'],
            video=runtime.config['train']['teacher_video_weight'],kv=runtime.config['train']['kv_weight']),
        disabled_objectives=['kv'] if runtime.config['train']['kv_weight']==0. else [],before_clipping=True,
        records=records,loss_rows=loss_records,loss_summary=scalar_summary(loss_records),
        loss_summary_scope='140 per-window measurements: 70 fixed fitting windows x 2 noise draws; not training-update ranges',
        elapsed_seconds=time.monotonic()-started)
    if runtime.rank==0:atomic_json(p,result)
    dist.barrier()
    return result


def sample_indices(length,device='cpu',limit=64):
    """Integer-only positions: float32 linspace can round length-1 to length."""
    if type(length) is not int or length<0 or type(limit) is not int or limit<1:
        raise ValueError('Invalid diagnostic sampling dimensions')
    count=min(length,limit)
    if count<2:return torch.arange(count,device=device,dtype=torch.long)
    if length-1>(2**63-1)//(count-1):raise ValueError('Diagnostic index multiplication would overflow')
    return torch.div(torch.arange(count,device=device,dtype=torch.long)*(length-1),count-1,rounding_mode='floor')


def sample_parameters(runtime):
    result={}
    with torch.no_grad():
        for name,p in runtime.engine.module.named_parameters():
            if not p.requires_grad:continue
            v=runtime.engine.optimizer.get_local_fp32_param(p).reshape(-1)
            n=v.numel()
            if not n:continue
            indices=sample_indices(n,v.device)
            result[name]=(v[indices].float().cpu().clone(),n)
    return result


def sampled_update(runtime,before):
    after=sample_parameters(runtime);groups=defaultdict(lambda:torch.zeros(3,dtype=torch.float64))
    for name,(a,n) in before.items():
        b,m=after[name]
        if n!=m:raise ValueError('Optimizer shard shape changed')
        a=a.double();b=b.double();scale=n/len(a)
        values=torch.tensor([float((b-a).square().sum())*scale,float(a.square().sum())*scale,n],dtype=torch.float64)
        groups[layer_name(name)]+=values;groups['all']+=values
    keys=sorted(groups);values=torch.stack([groups[k] for k in keys]).to(runtime.device)
    dist.all_reduce(values,op=dist.ReduceOp.SUM);out={}
    for name,(delta,base,n) in zip(keys,values.cpu().tolist()):
        out['optimizer_update_sampled/'+name+'/rms_estimate']=math.sqrt(delta/max(n,1))
        out['optimizer_update_sampled/'+name+'/relative_l2_estimate']=math.sqrt(delta/max(base,1e-30))
    return out
