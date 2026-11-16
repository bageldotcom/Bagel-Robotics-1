"""Fixed, episode-disjoint native ten-step action monitoring; no simulator claims."""
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import numpy as np
import torch
from .panel import digest,immutable,seed_for,file_hash
from .runtime import gather,check_frozen,model_versions
from .search_quality import quality_summary
from vpp2.critical_windows.inference import NativeInferenceReader,masked_prefill


def select_windows(panel,config):
    by_task=defaultdict(lambda:defaultdict(list))
    for r in panel['windows']:
        if r['role']=='monitor':by_task[r['task']][r['episode']].append(r)
    selected=[];count=config['train']['native_windows_per_task']
    for task in panel['tasks']:
        groups=by_task[task]
        if len(groups)<count:raise ValueError('Not enough independent native-monitor episodes')
        # Verified critical monitoring windows are kept; unknown phases stay unknown.
        critical=[r for rs in groups.values() for r in rs if (r.get('critical_steps') or 0)>0]
        chosen={}
        for r in sorted(critical,key=lambda r:(r['episode'],r['frame'])):
            if len(chosen)<count:chosen.setdefault(r['episode'],r)
        episodes=sorted(groups,key=lambda ep:seed_for(config['seed'],'native-monitor-episode',task,ep))
        for ep in episodes:
            if len(chosen)>=count:break
            if ep in chosen:continue
            rows=sorted(groups[ep],key=lambda r:r['frame'])
            index=seed_for(config['seed'],'native-monitor-window',task,ep)%len(rows)
            chosen[ep]=rows[index]
        selected.extend(sorted(chosen.values(),key=lambda r:r['episode']))
    return selected


def reader_manifest(rows):
    episodes=[]
    for r in rows:
        episodes.append(dict(episode=r['episode'],task=r['task'],suite=r['suite'],role='monitor',
            length=r['length'],path=r['episode_path'],files=r['source_files'],text=r['text'],
            text_sha256=r['text_sha256'],prompt=r['prompt']))
    return dict(episodes=episodes)


@contextmanager
def inference_mode(model,branch):
    # Native inference must not change the subsequent mask-training flags.
    modules=list(model.modules())+[branch]
    flags=[m.training for m in modules]
    flags_v=model.video_expert.use_gradient_checkpointing
    flags_a=model.action_expert.use_gradient_checkpointing
    try:
        model.eval();branch.eval()
        model.video_expert.use_gradient_checkpointing=False
        model.action_expert.use_gradient_checkpointing=False
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():yield
    finally:
        for module,flag in zip(modules,flags):module.training=flag
        model.video_expert.use_gradient_checkpointing=flags_v
        model.action_expert.use_gradient_checkpointing=flags_a


def evaluate_native(config,panel,model,branch,controller,update,rank,world):
    entries=select_windows(panel,config)
    signature=digest(dict(panel=panel['identity'],rows=[r['id'] for r in entries],settings=config['native_inference']))
    state=getattr(model,'_full_mask_native_monitor',None)
    if state is None:
        state=dict(identity=signature,reader=NativeInferenceReader(config,reader_manifest(entries)),reference={})
        model._full_mask_native_monitor=state
    if state['identity']!=signature:raise ValueError('Native monitoring panel changed')
    reader=state['reader'];hard=controller.gates(update,'binary');results=[]
    annotations={}
    if config.get('critical_annotations_file'):
        path=Path(config['critical_annotations_file'])
        if file_hash(path)!=config['critical_annotations_sha256']:raise ValueError('Verified phase annotations changed')
        annotations=json.loads(path.read_text())['annotations']
    versions=model_versions(model)
    with inference_mode(model,branch):
        for i,r in enumerate(entries):
            if i%world!=rank:continue
            inputs,target=reader.read(r['episode'],r['frame'])
            if r['id'] not in state['reference']:
                with masked_prefill(model,branch):reference=model.infer_action(**inputs)['action']
                state['reference'][r['id']]=reference.detach().cpu().clone()
            reference=state['reference'][r['id']]
            with masked_prefill(model,branch,*hard):prediction=model.infer_action(**inputs)['action']
            if prediction.shape!=(32,16) or not torch.isfinite(prediction).all():raise FloatingPointError('Invalid native generated actions')
            prediction=prediction.detach().cpu();target_tensor=torch.as_tensor(target['normalized'])
            count=min(24,target['valid_steps'])
            te=float((reference[:count].float()-target_tensor[:count]).square().mean())
            he=float((prediction[:count].float()-target_tensor[:count]).square().mean())
            phase_metrics=None
            annotation=annotations.get(str(r['episode']))
            if annotation is not None:
                from vpp2.critical_windows.contracts import critical_mask
                if annotation['role']!='monitor' or annotation['status']!='verified':raise ValueError('Unverified/fit annotation entered native monitor')
                phase_metrics={}
                for kind in [None]+sorted({x['kind'] for x in annotation['intervals']}):
                    mask=critical_mask(r['frame'],32,r['length'],annotation['intervals'],kind)
                    mask[count:]=False
                    n=int(mask.sum())
                    if n:
                        index=torch.as_tensor(mask)
                        phase_metrics['all_critical' if kind is None else kind]=dict(steps=n,
                            teacher_error=float((reference[index].float()-target_tensor[index]).square().mean()),
                            hard_error=float((prediction[index].float()-target_tensor[index]).square().mean()))
            results.append(dict(phase_metrics=phase_metrics,task=r['task'],suite=r['suite'],episode=r['episode'],frame=r['frame'],
                teacher_error=te,hard_error=he,executed_steps=count,phase_status=r.get('phase_status','unknown'),
                critical_annotation_scope=r.get('critical_annotation_scope','unknown'),
                reference_actions=reference.tolist(),masked_actions=prediction.tolist(),target_actions=target_tensor.tolist()))
    check_frozen(model,versions)
    rows=sorted(sum(gather(results),[]),key=lambda r:(r['task'],r['episode'],r['frame']))
    if len(rows)!=len(entries):raise ValueError('Incomplete native action monitoring')
    return dict(update=update,identity=signature,windows=len(rows),rows=rows,quality=quality_summary(rows),
        native_sampler=True,inference_steps=10,action_horizon=32,scored_prefix=24,
        ground_truth_is_policy_input=False,policy_retention_established=False,
        phase_labels_available=sum(r['phase_metrics'] is not None for r in rows),
        verified_critical_windows=sum(bool(r['phase_metrics']) for r in rows),
        metric='Normalized MSE of native generated action prefix on demonstrated histories; not a rollout evaluation.')
