"""Separate global mask churn from suite-conditioned action/sensitivity evidence."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from .data import collate,prepared
from .runtime import cache,predict,gather
from .controller import churn
from .search_quality import quality_summary,sensitivity_check
from vpp2.compact_kv.losses import action_error


def top(x,fraction=.05):
    return torch.argsort(x,descending=True,stable=True)[:max(1,round(x.numel()*fraction))].tolist()


def overlap(a,b):return len(set(a)&set(b))/max(1,len(set(a)|set(b)))


def evaluate(config,panel,model,branch,controller,reader,update,rank,world,full=False):
    entries=[r for r in panel['windows'] if r['role']=='monitor']
    if not full:
        per_episode=defaultdict(list)
        for r in entries:per_episode[(r['task'],r['episode'])].append(r)
        entries=[sorted(rs,key=lambda r:r['frame'])[key[1]%len(rs)] for key,rs in sorted(per_episode.items())]
    episode_windows=defaultdict(int)
    for r in entries:episode_windows[(r['task'],r['episode'])]+=1
    n_tasks=len(panel['tasks']);size=5120+39*13824
    gradients=torch.zeros(n_tasks,2,size,device='cuda',dtype=torch.float32)
    counts=torch.zeros(n_tasks,2,device='cuda');rows=[]
    hg,fg=controller.gates(update);hard=controller.gates(update,'binary')
    reference_cache=getattr(model,'_full_mask_flow_reference',None)
    if reference_cache is None:
        reference_cache={};model._full_mask_flow_reference=reference_cache
    for i,r in enumerate(entries):
        if i%world!=rank:continue
        # Fixed monitor inputs and noise across checkpoints; distinct from training.
        stratum=(r['temporal_quarter']+r['episode_order'])%4
        parts=prepared(model,collate(reader,[r],[-2_000_000-r['id']]),config['seed'],[stratum])
        with torch.no_grad():
            key=(panel['identity'],config['seed'],r['id'],stratum)
            if key not in reference_cache:
                tc=cache(branch,parts);teacher=predict(model,parts,tc);del tc
                reference_cache[key]=float(action_error(teacher,parts[5],parts[0]['action_loss_is_pad']))
                del teacher
            te=reference_cache[key]
            hc=cache(branch,parts,hard);binary=predict(model,parts,hc);del hc
            he=action_error(binary,parts[5],parts[0]['action_loss_is_pad'])
        # Differentiate with respect to gate VALUES, not saturated sigmoid logits.
        # Native model and optimizer remain untouched by this diagnostic pass.
        h=hg.detach().requires_grad_();f=fg.detach().requires_grad_()
        soft=predict(model,parts,cache(branch,parts,(h,f)))
        se=action_error(soft,parts[5],parts[0]['action_loss_is_pad'])
        gh,gf=torch.autograd.grad(se.mean(),(h,f))
        grad=torch.cat([gh.abs(),gf.abs().flatten()])
        if not torch.isfinite(grad).all():raise FloatingPointError('Nonfinite suite sensitivity')
        weight=1./episode_windows[(r['task'],r['episode'])]
        gradients[r['task_id'],r['audit_half']]+=grad*weight
        counts[r['task_id'],r['audit_half']]+=weight
        rows.append(dict(task=r['task'],suite=r['suite'],episode=r['episode'],frame=r['frame'],
                         teacher_error=float(te),hard_error=float(he),soft_error=float(se.detach()),
                         phase_status=r.get('phase_status','unknown'),
                         critical_annotation_scope=r.get('critical_annotation_scope','unknown'),
                         critical_steps=r.get('critical_steps')))
    if dist.is_initialized():
        dist.all_reduce(gradients);dist.all_reduce(counts)
    all_rows=sum(gather(rows),[])
    if not (counts>0).all():raise RuntimeError('Missing task/episode-half diagnostics')
    gradients=gradients/counts[:,:,None]
    task_suites={r['task']:r['suite'] for r in entries}
    by_task={}
    for task in panel['tasks']:
        rs=[r for r in all_rows if r['task']==task]
        episodes=sorted({r['episode'] for r in rs})
        by_task[task]={k:float(np.mean([np.mean([r[k] for r in rs if r['episode']==ep]) for ep in episodes])) for k in ('teacher_error','hard_error','soft_error')}
        by_task[task]['hard_relative_to_teacher']=by_task[task]['hard_error']/max(by_task[task]['teacher_error'],1e-8)
    suites={}
    for suite in sorted(set(task_suites.values())):
        tasks=[t for t in panel['tasks'] if task_suites[t]==suite]
        ids=[panel['tasks'].index(t) for t in tasks]
        evidence=gradients[ids].mean(0).cpu();g=evidence.mean(0)
        kept=torch.cat([hard[0],hard[1].flatten()]).cpu()
        h_top=top(g[:5120]);f_top=[top(x) for x in g[5120:].reshape(39,13824)]
        halves_h=overlap(top(evidence[0,:5120]),top(evidence[1,:5120]))
        halves_f=[overlap(top(a),top(b)) for a,b in zip(evidence[0,5120:].reshape(39,13824),evidence[1,5120:].reshape(39,13824))]
        group_means=[float(g[:5120].mean())]+[float(x.mean()) for x in g[5120:].reshape(39,13824)]
        floor=max(config['train'].get('sensitivity_absolute_floor',1e-12),max(group_means)*config['train'].get('sensitivity_relative_floor',1e-6))
        group_names=['hidden']+[f'ffn_{i:02d}' for i in range(39)]
        group_evidence=[dict(name=name,mean_gradient=mean,active=mean>floor,top=indices,half_overlap=agreement)
            for name,mean,indices,agreement in zip(group_names,group_means,[h_top]+f_top,[halves_h]+halves_f)]
        def retained(s,k):return float((s*k).sum()/s.sum().clamp_min(1e-20))
        suites[suite]=dict(hard_error=float(np.mean([by_task[t]['hard_error'] for t in tasks])),
            soft_error=float(np.mean([by_task[t]['soft_error'] for t in tasks])),
            teacher_error=float(np.mean([by_task[t]['teacher_error'] for t in tasks])),
            worst_task_ratio=max(by_task[t]['hard_relative_to_teacher'] for t in tasks),
            retained_hidden_sensitivity=retained(g[:5120],kept[:5120]),
            retained_ffn_sensitivity=[retained(s,k) for s,k in zip(g[5120:].reshape(39,13824),kept[5120:].reshape(39,13824))],
            hidden_top=h_top,ffn_top=f_top,half_hidden_overlap=halves_h,half_ffn_overlap=halves_f,
            sensitivity_groups=group_evidence,sensitivity_floor=floor,
            soft_to_binary_gap=float(np.mean([by_task[t]['hard_error']-by_task[t]['soft_error'] for t in tasks])))
    return dict(update=update,full=full,windows=len(all_rows),by_task=by_task,by_suite=suites,
                selection=controller.selection(),rows=all_rows,quality=quality_summary(all_rows),
                verified_critical_windows=sum((r.get('critical_steps') or 0)>0 for r in all_rows),
                sensitivity_scope='Absolute GT action gradient with respect to current soft gate values; task-balanced within each suite. This is a local diagnostic, not a guarantee of action retention.')


def compare(previous,current,config):
    cfg=config['train'];turn=churn(previous['selection'],current['selection']);suite={}
    for name,now in current['by_suite'].items():
        old=previous['by_suite'][name]
        evidence=sensitivity_check(old,now,config,overlap)
        change=abs(now['hard_error']-old['hard_error'])/max(old['hard_error'],1e-8)
        suite[name]=dict(hard_error_relative_change=change,sensitivity=evidence,
            stable=change<=cfg['suite_error_relative_change_max'] and evidence['passed'])
    stable=(turn['hidden']<=cfg['hidden_churn_max'] and max(turn['ffn'][:-1])<=cfg['ffn_churn_max'] and all(x['stable'] for x in suite.values()))
    return dict(mask_churn=turn,suite_evidence=suite,stable=stable,previous_update=previous.get('update',-1))


def confirm_full(previous,current,config):
    if previous is None:return dict(passed=False,reason='first_full_audit_sets_baseline')
    if not previous.get('full') or not current.get('full'):raise ValueError('Full confirmation requires full audits')
    comparison=compare(previous,current,config)
    cfg=config['train'];threshold=cfg['suite_sensitivity_top_overlap_min']
    if cfg.get('noise_aware_sensitivity',False):
        halves={name:bool(any(g['active'] for g in v['sensitivity_groups']) and all(not g['active'] or g['half_overlap']>=cfg['sensitivity_half_overlap_min'] for g in v['sensitivity_groups'])) for name,v in current['by_suite'].items()}
    else:
        halves={name:bool(v['half_hidden_overlap']>=threshold and min(v['half_ffn_overlap'])>=threshold) for name,v in current['by_suite'].items()}
    quality_ok=not cfg.get('early_stop_quality_check',False) or current['quality']['no_resolved_regressions']
    return dict(passed=comparison['stable'] and all(halves.values()) and quality_ok,
        comparison=comparison,independent_episode_halves=halves,action_quality_ok=quality_ok,
        reason='full_panel_stability_episode_half_and_action_quality_confirmation',policy_retention_established=False)
