"""Create a new 30k subset after the pilot review; leave V3 immutable."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from vpp2.full_mask.panel import digest,file_hash,immutable,seed_for
from vpp2.full_mask.curation import fingerprint_window
from vpp2.full_mask.curation_v3 import write_rows
from vpp2.full_mask.curation_v3_finalize import records
from .contracts import read_json,checked_identity,load_annotations,critical_mask,cue_visible
from .selection import choose,CoverageConflict,repair_candidate_grid



def validate_publication(config, original, rows, annotations, observed_anchors):
    """Verify all selection promises before publishing a readable panel."""
    selection=[r for r in rows if r['role']=='selection_initial']
    monitoring=[r for r in rows if r['role']=='monitor']
    expected_count=config['critical_selection']['target_total']
    if len(selection)!=expected_count:
        raise ValueError('Final selection count differs from the approved budget')
    def native_identity(r):
        return (r['episode'],r['frame'],r['content_sha256'],tuple(r['history_indices']))
    old_monitor=[r for r in original['windows'] if r['role']=='monitor']
    if sorted(map(native_identity,monitoring))!=sorted(map(native_identity,old_monitor)):
        raise ValueError('Monitoring window choices changed')
    expected={r['episode'] for r in original['windows'] if r['role']=='selection_initial'}
    if {r['episode'] for r in selection}!=expected:
        raise ValueError('Original fitting-episode coverage was not preserved')
    if set(r['episode'] for r in monitoring)&expected:
        raise ValueError('Fitting and monitoring episodes overlap')
    by_episode=defaultdict(list)
    for row in selection:by_episode[row['episode']].append(row)
    dense_steps=0;anchor_count=0
    for ep,group in by_episode.items():
        length=group[0]['length'];covered=np.zeros(length,dtype=bool)
        if len(group)>config['curation']['max_windows_per_episode']:
            raise ValueError(f'Episode {ep}: per-episode cap exceeded')
        for row in sorted(group,key=lambda r:r['frame']):
            start,end=row['action_interval']
            if start!=row['frame'] or end!=start+32 or start<0 or end>length:
                raise ValueError(f'Episode {ep}: incomplete action interval')
            if covered[start:end].any():raise ValueError(f'Episode {ep}: overlapping targets')
            covered[start:end]=True
        if ep in annotations:
            a=annotations[ep]
            if a['role']!='fit':raise ValueError('Monitoring annotation entered fitting data')
            required=critical_mask(0,length,length,a['intervals'])
            if (required & ~covered).any():
                raise ValueError(f'Episode {ep}: verified critical action steps were dropped')
            dense_steps+=int(required.sum())
        if ep in observed_anchors:
            a=observed_anchors[ep]
            if a['role']!='fit':raise ValueError('Monitoring anchors entered fitting data')
            if not covered[a['frames']].all():
                raise ValueError(f'Episode {ep}: observed task-relevant frame was dropped')
            anchor_count+=len(a['frames'])
    if set(observed_anchors)-expected:
        raise ValueError('Observed fitting episode absent from selection')
    if len({r['content_sha256'] for r in rows})!=len(rows):
        raise ValueError('Duplicate content within or between selection and monitoring')
    selected_tasks=Counter(r['task'] for r in selection)
    if selected_tasks!=Counter(r['task'] for r in original['windows'] if r['role']=='selection_initial'):
        raise ValueError('Per-task selection budgets changed')
    for task,count in selected_tasks.items():
        random_count=sum(r['task']==task and r['selection_reason']=='balanced_random_coverage' for r in selection)
        if random_count!=round(count*config['critical_selection']['random_fraction']):
            raise ValueError(f'Task {task}: reserved random coverage changed')
    return dict(passed=True,selection_windows=len(selection),monitor_windows=len(monitoring),
                fitting_episodes=len(expected),verified_critical_action_steps=dense_steps,
                observed_task_relevant_frames=anchor_count,action_target_overlap_count=0,
                unchanged_monitoring_choices=True,unchanged_task_budgets=True,
                exact_duplicate_count=0)


def build(config):
    root=Path(config['root']);base=Path(config['base_data'])
    manifest=checked_identity(read_json(root/'audit_manifest.json'))
    annotations,receipts=load_annotations(config,manifest)
    staged=config.get('review_protocol')=='staged_visual_v1'
    observed_anchors={};screen=None
    if staged:
        from .staged_review import assess
        screen,observed_anchors=assess(config)
        if not screen['selection_ready']:
            raise ValueError(f"Staged screen incomplete or has unresolved data defects: {screen['missing_episodes']}")
    elif len(annotations)!=len(manifest['episodes']):
        raise ValueError(f'Pilot visual review is incomplete: {len(annotations)}/{len(manifest["episodes"])}; no revised panel published')
    original=checked_identity(read_json(base/'panel.json'));inv=checked_identity(read_json(base/'inventory.json'))
    if original['identity']!=manifest['base_panel_identity'] or inv['identity']!=manifest['inventory_identity']:
        raise ValueError('Base artifacts changed')
    quality=read_json(base/'quality_receipt.json')
    if file_hash(base/'quality_models.pt')!=quality['model_sha256']:raise ValueError('Quality helper changed')
    models=torch.load(base/'quality_models.pt',map_location='cpu',weights_only=False)
    scalers=read_json(base/'descriptor_scalers.json')
    base_config=read_json(config['base_config'])
    old={(r['episode'],r['frame']):r for r in original['windows']}
    fit_ids={e['episode'] for e in manifest['episodes'] if e['role']=='fit'}
    selection=[];task_reports={};annotation_identity=digest(dict(fit_reviews=[r for r in receipts if r['episode'] in fit_ids],screen=screen['record_identities'] if screen else None,protocol=screen['stage_identity'] if screen else None))
    builder_identity=digest(dict(annotation_identity=annotation_identity,selection=config['critical_selection'],cap=config['curation']['max_windows_per_episode'],seed=config['seed'],source={name:file_hash(Path(__file__).parent/name) for name in ('reselect.py','selection.py')}))
    output=root/'revisions'/builder_identity[:16];output.mkdir(parents=True,exist_ok=True)
    try:
        for task in inv['tasks']:
            episodes=[e for e in inv['episodes'] if e['task']==task and e['role']=='fit']
            scaler={k:tuple(np.array(v) for v in pair) for k,pair in scalers[task].items()}
            candidates,x=records(base_config,episodes,'selection_initial',scaler,quality,models)
            relevant={ep:a for ep,a in annotations.items() if a['task']==task and a['role']=='fit'}
            task_anchors={ep:a for ep,a in observed_anchors.items() if ep in {e['episode'] for e in episodes}}
            candidates,x,plans,repairs=repair_candidate_grid(candidates,x,base_config['curation']['max_windows_per_episode'],relevant,task_anchors)
            indices,reasons=choose(candidates,x,inv['quotas'][task],seed_for(config['seed'],'v3-select',task),
                                   base_config['curation']['max_windows_per_episode'],relevant,
                                   base_config['curation']['base_windows_per_episode'],config['critical_selection']['random_fraction'],
                                   observed_anchors=task_anchors,preplanned_windows=plans)
            current=[]
            for i in indices:
                row=dict(candidates[i],selection_reason=reasons[i]);a=relevant.get(row['episode'])
                if a is not None:
                    row.update(critical_annotation_scope='verified_pilot_episode',
                               critical_steps=critical_mask(row['frame'],32,row['length'],a['intervals']).tolist(),
                               critical_interval_ids=[v['id'] for v in a['intervals'] if v['frames'][0]<row['frame']+32 and v['frames'][1]>row['frame']])
                else:
                    row.update(critical_annotation_scope='not_annotated',critical_steps=None,critical_interval_ids=None)
                previous=old.get((row['episode'],row['frame']))
                row['content_sha256']=previous['content_sha256'] if previous else fingerprint_window(base_config,row)
                current.append(row)
            if len({r['content_sha256'] for r in current})!=len(current):raise CoverageConflict('Duplicate content requires explicit replacement')
            selection.extend(current)
            task_reports[task]=dict(selected=len(current),verified_fit_episodes=len(relevant),
                                    unknown_fit_episodes=len(episodes)-len(relevant),candidate_grid_repairs=repairs,reasons=dict(Counter(r['selection_reason'] for r in current)))
            print(json.dumps(dict(event='critical_task_selected',task=task,windows=len(current))),flush=True)
    except CoverageConflict as exc:
        immutable(output/'coverage_conflict.json',dict(annotation_identity=annotation_identity,error=str(exc),published=False))
        raise
    if len(selection)!=30000:raise ValueError('Unexpected final sample count')
    monitoring=[dict(r) for r in original['windows'] if r['role']=='monitor']
    coverage={}
    for task in inv['tasks']:
        selected=[r for r in selection if r['task']==task]
        previous=original['coverage'][task]
        coverage[task]={**previous,'selected_windows':len(selected),
                        'selected_episodes':len({r['episode'] for r in selected}),
                        'windows_per_episode':dict(Counter(r['episode'] for r in selected)),
                        'temporal_quarters':dict(Counter(r['temporal_quarter'] for r in selected)),
                        'proxy_tags':dict(Counter(t for r in selected for t in r['proxy_tags'])),
                        'reasons':dict(Counter(r['selection_reason'] for r in selected)),
                        'random_fraction_actual':sum(r['selection_reason']=='balanced_random_coverage' for r in selected)/len(selected),
                        'monitor_descriptor_distance':None,'phase_verified':False,
                        'verified_fit_episodes':task_reports[task]['verified_fit_episodes'],
                        'verification_scope':'Pilot episodes only; whole-task verification is not claimed.'}
    rows=selection+monitoring
    publication_checks=validate_publication(config,original,rows,annotations,observed_anchors)
    for i,r in enumerate(rows):
        r['id']=i;r['window_id']=digest(['critical-v1',manifest['source_identity'],r['episode'],r['frame'],r['role'],annotation_identity])
    panel={**{k:v for k,v in original.items() if k not in ('identity','windows','coverage','limitations')},
           'schema':'robodojo-critical-calibration-v1','windows':rows,'parent_panel_identity':original['identity'],
           'annotation_identity':annotation_identity,'builder_identity':builder_identity,'critical_coverage':task_reports,
           'review_protocol':config.get('review_protocol','exhaustive_280_v1'),
           'screen_stage_identity':screen['stage_identity'] if screen else None,
           'coverage':coverage,
           'limitations':[f"Dense annotations: {len(annotations)} development episodes; coarse fitting screens: {len(observed_anchors)}.",
                          'Unannotated episodes are explicitly unknown, not certified noncritical.',
                          'Mask training requires fresh data/provenance/native checks for this panel.'],
           'training_ready':False}
    panel['identity']=digest(panel)
    if staged:
        receipt,_=assess(config,panel)
        if not receipt['passed']:raise ValueError('Revised panel did not retain every visually observed anchor')
    immutable(output/'panel.json',panel)
    write_rows(output/'selection_manifest.jsonl',selection)
    # Retain the unmodified monitoring window choices. Their dense labels are
    # used only for metrics, never fitting-window selection.
    write_rows(output/'monitor_manifest.jsonl',monitoring)
    if staged:
        from .staged_review import write_status
        receipt=write_status(config,panel)
        immutable(output/'visual_review_receipt.json',receipt)

    immutable(output/'receipt.json',dict(panel_identity=panel['identity'],selection_windows=len(selection),
              monitored_annotations_used_for_selection=False,task_reports=task_reports,publication_checks=publication_checks,training_launched=False))
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();print(build(read_json(a.config)))


if __name__=='__main__':main()
