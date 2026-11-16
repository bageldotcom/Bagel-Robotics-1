"""Build the reviewable subset; final-audit episode content remains sealed."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
import torch
from .curation_v3 import SCHEMA,choose,write_rows
from .curation_v3_quality import raw_episode,normalized
from .curation import history_indices,robust_transform,transform,GROUPS,fingerprint_window
from .panel import digest,immutable,file_hash,seed_for


def records(config,episodes,role,scaler,quality,models):
    out=Path(config['curation']['output']);rows=[];matrices=[]
    projection=np.random.default_rng(config['seed']).normal(size=(2592,96)).astype(np.float32)/np.sqrt(2592)
    for order,e in enumerate(episodes):
        c,obs,actions=raw_episode(out,e)
        m=transform(c['motion'],*scaler['motion'])
        aa=transform(actions,*scaler['action'])
        vis=normalized(obs,models['mean'][0],models['scale'][0])[:,:-16]@projection
        blocks=[m[:,ix]/np.sqrt(len(ix)) for ix in GROUPS.values()]
        blocks.extend([aa/np.sqrt(aa.shape[1]),vis/np.sqrt(vis.shape[1])])
        x=np.concatenate(blocks,axis=-1).astype(np.float32)/np.sqrt(len(blocks))
        q=quality['episodes'].get(str(e['episode']),dict(rank=.5,usable=False))
        for j,t in enumerate(c['starts']):
            t=int(t);tags=[]
            if c['motion'][j,4:6].max()>.1:tags.append('gripper_transition_candidate')
            if c['motion'][j,14:16].sum()>1e-5:tags.append('slow_rotation_candidate')
            if c['motion'][j,10]>.1:tags.append('both_arm_motion_candidate')
            rows.append(dict(task=e['task'],task_id=e['task_id'],suite=e['suite'],episode=e['episode'],frame=t,
                length=e['length'],episode_path=e['path'],episode_order=order,role=role,
                text=e['text'],text_sha256=e['text_sha256'],prompt=e['prompt'],history_indices=history_indices(t),
                action_interval=[t,t+32],source_files=e['files'],temporal_quarter=min(3,t*4//max(1,e['length']-31)),
                relative_time=t/max(1,e['length']-32),proxy_tags=tags,protected_suite=e['suite'] in ('memory','precision'),
                phase_status='unverified_event_candidate',memory_cue_observable=None if e['suite']=='memory' else 'not_applicable',
                quality_rank=q['rank'],quality_usable=q['usable'],descriptor_row=j,audit_half=order%2))
        matrices.append(x)
    return rows,np.concatenate(matrices)


def build(config):
    out=Path(config['curation']['output']);inv=json.loads((out/'inventory.json').read_text())
    quality=json.loads((out/'quality_receipt.json').read_text())
    if quality['inventory_identity']!=inv['identity'] or file_hash(out/'quality_models.pt')!=quality['model_sha256']:
        raise ValueError('Quality artifact identity mismatch')
    models=torch.load(out/'quality_models.pt',map_location='cpu',weights_only=False)
    selection=[];monitor=[];reserve=[];near=[];dedup=[];coverage={};scalers={};all_fingerprints={}
    for task in inv['tasks']:
        episodes=[e for e in inv['episodes'] if e['task']==task and e['role']=='fit']
        held=[e for e in inv['episodes'] if e['task']==task and e['role']=='monitor']
        cs=[];ac=[]
        for e in episodes:
            with np.load(out/'candidates'/f"{e['episode']:05d}.npz",allow_pickle=False) as z:
                cs.append(z['motion']);ac.append(z['action'])
        scaler=dict(motion=robust_transform(np.concatenate(cs)),action=robust_transform(np.concatenate(ac)))
        scalers[task]={k:[v.tolist() for v in values] for k,values in scaler.items()}
        rows,x=records(config,episodes,'selection_initial',scaler,quality,models)
        selected,reasons,distances=choose(rows,x,inv['quotas'][task],seed_for(config['seed'],'v3-select',task),
            config['curation']['max_windows_per_episode'],config['curation']['base_windows_per_episode'],config['curation']['random_fraction'])
        kept=[];kept_indices=[]
        for i in selected:
            r=dict(rows[i],selection_reason=reasons[i]);fp=fingerprint_window(config,r);r['content_sha256']=fp
            if fp in all_fingerprints:
                dedup.append(dict(task=task,episode=r['episode'],frame=r['frame'],representative=all_fingerprints[fp],reason='identical_native_context_and_ground_truth_actions'))
                continue
            all_fingerprints[fp]=[r['episode'],r['frame']];kept.append(r);kept_indices.append(i)
        # Do not force replacement if it would violate actual coverage or nonoverlap.
        selected_keys={(r['episode'],r['frame']) for r in kept};selection.extend(kept)
        for r in rows:
            if (r['episode'],r['frame']) not in selected_keys:
                reserve.append(dict(r,role='reserve',shares_selected_episode=True,eligible_for_automatic_expansion=False))
        hr,hx=records(config,held,'monitor',scaler,quality,models)
        goal=sum(min(4,e['length']//32) for e in held)
        hi,hreason,_=choose(hr,hx,goal,seed_for(config['seed'],'v3-monitor',task),4,base=4,random_fraction=0)
        hm=[]
        for i in hi:
            r=dict(hr[i],selection_reason='development_monitor_temporal_coverage');r['content_sha256']=fingerprint_window(config,r)
            if r['content_sha256'] in all_fingerprints:raise ValueError('Native content crosses selection/development split')
            hm.append(r)
        monitor.extend(hm)
        # Restrict expensive pair comparisons to small blocks and do not auto-delete.
        sx=x[kept_indices];nearest=[]
        for start in range(0,len(sx),128):
            d=cdist(sx[start:start+128],sx)
            for j in range(len(d)):d[j,start+j]=np.inf
            nearest.extend((float(d[j].min()),start+j,int(d[j].argmin())) for j in range(len(d)))
        pairs=set()
        for distance,i,j in sorted(nearest)[:20]:
            pair=tuple(sorted((i,j)))
            if pair in pairs:continue
            pairs.add(pair);a=rows[kept_indices[i]];b=rows[kept_indices[j]]
            near.append(dict(task=task,first=[a['episode'],a['frame']],second=[b['episode'],b['frame']],
                             descriptor_distance=distance,decision='retain_pending_context_review',
                             reason='similarity_does_not_prove_equivalent_manipulation'))
        dh=np.concatenate([cdist(hx[i:i+128],sx).min(-1) for i in range(0,len(hx),128)])
        coverage[task]=dict(suite=episodes[0]['suite'],selected_windows=len(kept),target=inv['quotas'][task],
            selected_episodes=len({r['episode'] for r in kept}),available_fit_episodes=len(episodes),
            monitor_windows=len(hm),monitor_episodes=len(held),audit_episodes=8,
            windows_per_episode=dict(Counter(r['episode'] for r in kept)),
            temporal_quarters=dict(Counter(r['temporal_quarter'] for r in kept)),
            proxy_tags=dict(Counter(tag for r in kept for tag in r['proxy_tags'])),
            reasons=dict(Counter(r['selection_reason'] for r in kept)),
            quality_rank_usable=quality['task_checks'][task]['ranking_usable'],
            monitor_descriptor_distance=float(np.mean(dh)),phase_verified=False,
            cue_observability_verified=False if episodes[0]['suite']=='memory' else 'not_applicable',
            random_fraction_actual=sum(r['selection_reason']=='balanced_random_coverage' for r in kept)/len(kept))
        print(json.dumps(dict(event='task_selected_v3',task=task,windows=len(kept),episodes=coverage[task]['selected_episodes'])),flush=True)
    allrows=selection+monitor
    for i,r in enumerate(allrows):
        r['id']=i;r['window_id']=digest([SCHEMA,inv['source_identity'],r['episode'],r['frame'],r['role']])
    for i,r in enumerate(reserve):
        r['id']=len(allrows)+i;r['window_id']=digest([SCHEMA,inv['source_identity'],r['episode'],r['frame'],'reserve'])
    panel=dict(schema=SCHEMA,seed=config['seed'],source=config['demonstrations'],source_identity=inv['source_identity'],
        source_manifest_sha256=inv['source_manifest_sha256'],normalization_sha256=inv['normalization_sha256'],
        training_split_sha256=inv['training_split_sha256'],validation_split_sha256=inv['validation_split_sha256'],
        inventory_identity=inv['identity'],quality_receipt_sha256=file_hash(out/'quality_receipt.json'),
        visual_receipt_sha256=file_hash(out/'visual_features_receipt.json'),curation_config=config['curation'],
        tasks=inv['tasks'],windows=allrows,coverage=coverage,counts=dict(Counter(r['role'] for r in allrows)),
        official_validation_episodes=inv['official_validation_episodes'],
        episode_partitions=[dict(task=e['task'],episode=e['episode'],role=e['role']) for e in inv['episodes']],
        limitations=['Contact, manipulation stages and memory-cue visibility remain unverified until visual review.',
            'Quality ranks are DemInf-inspired tie-breakers, not success probabilities.',
            'The final 280 audit episodes are not read by feature/quality/selection stages.',
            'Thirty thousand is a requested coverage target, not a sufficiency guarantee.'],
        open_suite='No Open-suite training demonstrations; DLC is auxiliary, not Open.')
    panel['identity']=digest(panel);immutable(out/'panel.json',panel)
    for name,rs in [('selection_manifest.jsonl',selection),('monitor_manifest.jsonl',monitor),('reserve_manifest.jsonl',reserve),
                    ('redundancy_audit.jsonl',dedup),('near_duplicate_review.jsonl',near)]:write_rows(out/name,rs)
    immutable(out/'coverage.json',dict(panel_identity=panel['identity'],by_task=coverage))
    immutable(out/'descriptor_scalers.json',scalers)
    immutable(out/'audit_episode_manifest.json',dict(panel_identity=panel['identity'],locked=True,
        episodes=[e for e in inv['episodes'] if e['role']=='audit'],unseal_requires='Explicit final-mask audit invocation after mask checkpoint is fixed'))
    immutable(out/'review_requirements.json',dict(panel_identity=panel['identity'],status='pending',
        required=['Inspect task-specific manipulation phases and Precision/Memory examples.',
                  'Verify proposed redundancy removals against native history and actions.',
                  'Pass native GPU invariance and action-sensitivity coverage diagnostics.'],
        automatic_near_duplicate_deletions=0,exact_duplicates_removed=len(dedup)))
    print(json.dumps(dict(event='curated_v3_complete',identity=panel['identity'],counts=panel['counts'])),flush=True)
    return panel
