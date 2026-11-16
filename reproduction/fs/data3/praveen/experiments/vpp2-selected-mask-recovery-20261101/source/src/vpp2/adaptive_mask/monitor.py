"""Fixed monitor windows; paired actions/video, native sampled actions and score utility."""
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import json
import numpy as np
import torch
from vpp2.full_mask.panel import seed_for,digest
from vpp2.full_mask.native_monitor import select_windows,reader_manifest,inference_mode
from vpp2.critical_windows.inference import NativeInferenceReader,masked_prefill
from .runtime import forward_video,predict,gather,step
from .data import prepare,future_targets,future_mask
from .losses import video_error,component_errors
from vpp2.compact_kv.losses import action_error
from .quality import assess
from .quality_policy import quality_flags, monitor_acceptance, policy_mode
from .feedback import diagnostic_rows
from .state import diagnostic_rng,save

@contextmanager
def native_context(model,branch,gates=(None,None,None)):
    def cache_only(inputs,noise,t,mask,*unused):return branch(inputs,noise,t,mask,*gates,return_video=False)
    with masked_prefill(model,cache_only):yield

def monitor_rows(panel,full):
    rows=[r for r in panel['windows'] if r['role']=='monitor']
    if full:return rows
    groups=defaultdict(list)
    for row in rows:groups[row['task']].append(row)
    selected=[]
    for task,group in sorted(groups.items()):
        episodes=set()
        for row in sorted(group,key=lambda r:seed_for(100008,'routine-monitor',r['id'])):
            if row['episode'] not in episodes:selected.append(row);episodes.add(row['episode'])
            if len(episodes)==4:break
    return selected

def summarize(rows,parent=None,quality_config=None):
    settings=quality_config or {}
    reference=assess(rows,**settings)
    matched=[]
    if parent is not None:
        lookup={(r['task'],r['episode'],r['frame'],r.get('stratum',-1)):r for r in parent}
        for row in rows:
            key=(row['task'],row['episode'],row['frame'],row.get('stratum',-1))
            if key not in lookup:raise ValueError('Parent quality panel mismatch')
            matched.append(dict(row,parent_error=lookup[key]['student_error']))
        previous=assess(matched,reference_key='parent_error',**settings)
    else:previous=reference
    return dict(original=reference,parent=previous,passed=reference['passed'] and previous['passed'],
        clearly_passing=reference['clearly_passing'] and previous['clearly_passing'],
        resolved_violation=reference['resolved_violation'] or previous['resolved_violation'])



def component_quality(rows, parent_rows, settings):
    names=('position','quaternion_coordinates','gripper')
    reports={}
    for index,name in enumerate(names):
        def select(values):
            return [dict(r,teacher_error=r['teacher_components'][index],student_error=r['student_components'][index]) for r in values]
        reports[name]=summarize(select(rows),None if parent_rows is None else select(parent_rows),settings)
    return reports


def quality_settings(config):
    # Keep the original nine-family correction even when video is diagnostic.
    # This protocol revision does not relax action margins or confidence bounds.
    # These repeated development decisions are screening rules, not a final CI.
    settings=dict(config['quality']);settings['alpha']=settings.get('alpha',.05)/18
    return settings

def evaluate(config,panel,model,reader,branch,controller,pruning_updates,rank,world,full=False,parent=None,rows_override=None):
    rows=monitor_rows(panel,full) if rows_override is None else rows_override;gates=controller.forward_gates(mode='binary');results=[];video_rows=[];examples=[]
    with diagnostic_rng(),torch.no_grad():
        for i,row in enumerate(rows):
            if row.get('owner_index',i)%world!=rank:continue
            s=row.get('audit_stratum',seed_for(config['seed'],'monitor-stratum',row['id'])%4)
            parts=prepare(model,reader,[row],[-2_000_000-row['id']],config['seed'],[s])
            teacher=forward_video(branch,parts);student=forward_video(branch,parts,gates)
            ta=predict(model,parts,teacher['cache']);sa=predict(model,parts,student['cache'])
            from .losses import objectives
            from .losses import cache_terms
            loss,metrics,details,_=objectives(sa,ta,parts[5],parts[0]['action_loss_is_pad'],model.train_action_scheduler.training_weight(parts[4]),
                student['video'],teacher['video'],cache_terms(student['cache'],teacher['cache']),future_mask([row],parts[1].device),
                pruning_updates)
            target_path=Path(config['root'])/'monitor_targets'/f"{row['id']:08d}.pt"
            if target_path.exists():
                target_state=torch.load(target_path,map_location=model.device,weights_only=True)
                if target_state['panel_identity']!=panel['identity']:raise ValueError('Wrong monitor target identity')
                target=target_state['target']
            else:
                target=future_targets(model,reader,[row]);save(target_path,dict(panel_identity=panel['identity'],target=target.cpu()))
            truth=model.train_video_scheduler.training_target(target,parts[1],parts[2])
            se,valid=video_error(student['video'],truth,future_mask([row],parts[1].device))
            te,_=video_error(teacher['video'],truth,future_mask([row],parts[1].device))
            result=dict(window_id=row['id'],task=row['task'],suite=row['suite'],episode=row['episode'],frame=row['frame'],stratum=int(s),
                student_error=float(details['student_error']),teacher_error=float(details['teacher_error']),
                student_components=details['student_components'][0].tolist(),teacher_components=details['teacher_components'][0].tolist(),
                base_reliability=float(details['base_reliability']),reliability=float(details['reliability']),
                student_veto=bool(details['student_veto']),zero_veto=bool(details['zero_veto']),component_veto=bool(details['component_veto']),
                video_distillation_error=float(details['video_error']),
                kv=[[float(metrics[f'kv/layer_{l:02d}_{k}']) for k in ('k','v')] for l in range(40)],
                critical_annotation_scope=row.get('critical_annotation_scope','not_annotated'),phase_status=row.get('phase_status','unverified'))
            results.append(result)
            if bool(valid):video_rows.append(dict(result,student_error=float(se),teacher_error=float(te)))
            if i<5:
                examples.append(dict(window_id=row['id'],task=row['task'],teacher=ta.float().cpu().tolist(),student=sa.float().cpu().tolist(),target=parts[5].float().cpu().tolist(),
                    teacher_future=(parts[1]-teacher['video'])[:,:,3:].float().square().mean(1).sqrt().cpu().tolist(),
                    student_future=(parts[1]-student['video'])[:,:,3:].float().square().mean(1).sqrt().cpu().tolist(),
                    label='Single-step predicted clean-latent magnitude; not sampled RGB video'))
            if len(results)%64==0:print(json.dumps(dict(event='monitor',full=full,rank=rank,done=len(results))),flush=True)
    results=sorted(sum(gather(results),[]),key=lambda r:r['window_id'])
    video_rows=sorted(sum(gather(video_rows),[]),key=lambda r:r['window_id'])
    examples=sum(gather(examples),[])
    if len(results)!=len(rows) or len({r['window_id'] for r in results})!=len(rows):raise ValueError('Incomplete monitor')
    settings=quality_settings(config)
    action=summarize(results,None if parent is None else parent['rows'],settings)
    video=summarize(video_rows,None if parent is None else parent['video_rows'],settings)
    components=component_quality(results,None if parent is None else parent['rows'],settings)
    return dict(rows=results,video_rows=video_rows,examples=examples,full=full,action_quality=action,video_quality=video,
        component_quality=components,**monitor_acceptance(config,action,video,components))


def native_evaluate(config,panel,model,branch,controller,rank,world,parent=None,entries_override=None):
    entries=select_windows(panel,config) if entries_override is None else entries_override
    key=digest(dict(panel=panel['identity'],entries=[r['id'] for r in entries],inference=config['native_inference']))
    state=getattr(model,'_adaptive_native_monitor',None)
    if state is None:
        state=dict(key=key,reader=NativeInferenceReader(config,reader_manifest(entries)),teacher={})
        model._adaptive_native_monitor=state
    if state['key']!=key:raise ValueError('Native monitor inputs changed')
    reader=state['reader'];results=[];examples=[]
    gates=controller.forward_gates(mode='binary')
    with diagnostic_rng(),inference_mode(model,branch):
        for i,row in enumerate(entries):
            if row.get('owner_index',i)%world!=rank:continue
            inputs,target=reader.read(row['episode'],row['frame'])
            if row['id'] not in state['teacher']:
                with native_context(model,branch):state['teacher'][row['id']]=model.infer_action(**inputs)['action'].float().cpu().clone()
            ta=state['teacher'][row['id']]
            with native_context(model,branch,gates):sa=model.infer_action(**inputs)['action'].float().cpu()
            gt=torch.as_tensor(target['normalized']);n=min(target['valid_steps'],24)
            if n<1:raise ValueError('No valid native action steps')
            tc=component_errors(ta[:n].unsqueeze(0),gt[:n].unsqueeze(0),None)[0].tolist()
            sc=component_errors(sa[:n].unsqueeze(0),gt[:n].unsqueeze(0),None)[0].tolist()
            results.append(dict(window_id=row['id'],task=row['task'],suite=row['suite'],episode=row['episode'],frame=row['frame'],
                teacher_error=float((ta[:n]-gt[:n]).square().mean()),student_error=float((sa[:n]-gt[:n]).square().mean()),
                teacher_components=tc,student_components=sc))
            if i<5:examples.append(dict(task=row['task'],window_id=row['id'],teacher=ta.tolist(),student=sa.tolist(),target=gt.tolist(),executed_prefix=n))
    results=sorted(sum(gather(results),[]),key=lambda r:r['window_id']);examples=sum(gather(examples),[])
    if len(results)!=len(entries):raise ValueError('Missing native sampled-action results')
    settings=quality_settings(config)
    quality=summarize(results,None if parent is None else parent['rows'],settings)
    components=component_quality(results,None if parent is None else parent['rows'],settings)
    return dict(rows=results,examples=examples,quality=quality,component_quality=components,
                **quality_flags([quality,*components.values()]),quality_gate_mode=policy_mode(config),
                native_10_step_sampler=True,simulation=False)


def gradient_diagnostics(config,panel,model,reader,branch,controller,pruning_updates,rank,world,full=False,check_index=0):
    rows=diagnostic_rows(panel,config['seed'],check_index);records=[]
    families=[('hidden',controller.hidden_scores,0),('ffn',controller.ffn_scores,1),('cross_heads',controller.cross_scores,2)]
    families=[x for x in families if x[1].requires_grad]
    params=[x[1] for x in families]
    with diagnostic_rng():
        for i,row in enumerate(rows):
            if i%world!=rank:continue
            if row['role']!='selection_initial':raise ValueError('Gradient diagnostics must use fitting data')
            suite=row['suite']
            parts=prepare(model,reader,[row],[-3_000_000-row['id']],config['seed'],[2])
            loss,metrics,details,terms=step(model,branch,controller,parts,[row],pruning_updates,config)
            keys=['gt_action','late_kv','teacher_video'] if full else ['gt_action'];vectors={}
            for j,key in enumerate(keys):
                grad=torch.autograd.grad(terms[key],params,retain_graph=j<len(keys)-1,allow_unused=False)
                vectors[key]=[x.detach() for x in grad]
            record=dict(suite=suite,task=row['task'],window_id=row['id'],data_role='selection_initial',check_index=check_index,families={})
            for j,(family,parameter,gate_index) in enumerate(families):
                values={key:float(vectors[key][j].norm()) for key in keys}
                g=vectors['gt_action'][j];hard=controller.gates(mode='binary')[gate_index]
                values['retained_gt_sensitivity']=float((g.abs()*hard).sum()/g.abs().sum().clamp_min(1e-20))
                values['score_derivative_mean']=float((params[j].sigmoid()*(1-params[j].sigmoid())).mean())
                values['active']=bool(torch.isfinite(g).all() and g.norm()>1e-12)
                if full:
                    for key,coefficient in [('late_kv',.05),('teacher_video',.05)]:
                        auxiliary=vectors[key][j]
                        values[key+'_weighted_norm']=float(auxiliary.norm()*coefficient*min(pruning_updates/100,1))
                        values[key+'_cosine_with_gt']=float((g*auxiliary).sum()/(g.norm()*auxiliary.norm()).clamp_min(1e-20))
                record['families'][family]=values
            record['protected_hidden_width']=not controller.hidden_scores.requires_grad
            records.append(record)
    return sum(gather(records),[])
