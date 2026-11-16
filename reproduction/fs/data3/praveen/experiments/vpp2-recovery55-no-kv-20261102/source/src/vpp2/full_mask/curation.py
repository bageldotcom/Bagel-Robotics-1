"""Training-only, nonoverlapping calibration selection.

Geometry and motion are measured; contact/cue semantics are NOT inferred as truth.
RGB tiles are a conservative visual-layout proxy, not a learned semantic encoder.
Near-duplicate suggestions never delete examples without a review decision.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .panel import digest, file_hash, immutable, seed_for

SCHEMA = 'robodojo-curated-calibration-v2'
HORIZON = 32
FEATURE_VERSION = 'physical-motion-and-ordered-rgb-tiles-v2'
MOTION_NAMES = [
    'left_path', 'right_path', 'left_rotation', 'right_rotation',
    'left_grip_variation', 'right_grip_variation', 'left_displacement', 'right_displacement',
    'left_active_fraction', 'right_active_fraction', 'both_active_fraction',
    'relative_hand_distance_change', 'left_turning', 'right_turning',
    'left_slow_rotation', 'right_slow_rotation',
    'left_grip_mean', 'right_grip_mean',
    'left_x', 'left_y', 'left_z', 'right_x', 'right_y', 'right_z',
    'left_initial_grip', 'right_initial_grip',
    'left_state_action_gap', 'right_state_action_gap',
]
GROUPS = {'translation': list(range(0, 2)) + list(range(6, 8)) + list(range(12, 14)),
          'rotation': [2, 3, 14, 15], 'grip': [4, 5, 16, 17, 24, 25],
          'two_arm': [8, 9, 10, 11], 'state': list(range(18, 24)) + [26, 27]}


def history_indices(t):
    return [0] + [max(0, int(t) + o) for o in range(-175, 1, 25)]


def candidate_starts(length, stride=8):
    if length < HORIZON:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.r_[np.arange(0, length - HORIZON + 1, stride), length - HORIZON]).astype(np.int64)


def valid_actions(a):
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[1] != 16 or not np.isfinite(a).all():
        raise ValueError('Require finite raw [T,16] EE actions/states')
    for start in (3, 11):
        if np.any(np.linalg.norm(a[:, start:start+4], axis=-1) < 1e-8):
            raise ValueError('Zero-norm quaternion')
    return a


def quaternion_steps(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return np.r_[0., 2 * np.arccos(np.clip(np.abs((q[1:] * q[:-1]).sum(-1)), 0, 1))]


def motion_descriptors(actions, states, stride=8):
    a = valid_actions(actions).astype(np.float64)
    s = valid_actions(states).astype(np.float64)
    if a.shape != s.shape:
        raise ValueError('Action/state length mismatch')
    xyz = a[:, [0,1,2,8,9,10]].reshape(-1,2,3)
    dp = np.linalg.norm(np.diff(xyz, axis=0, prepend=xyz[:1]), axis=-1)
    rot = np.stack([quaternion_steps(a[:,3:7]), quaternion_steps(a[:,11:15])], -1)
    grip = np.abs(np.diff(a[:,[7,15]], axis=0, prepend=a[:1,[7,15]]))
    active = (dp > .0005) | (rot > np.deg2rad(.5)) | (grip > .02)
    starts = candidate_starts(len(a), stride)
    # Add event-centred candidates; these are command transitions, not contacts.
    event = np.flatnonzero((grip.max(-1) > .1) | (rot.max(-1) > np.quantile(rot.max(-1), .95)))
    if len(event):
        starts = np.unique(np.r_[starts, np.clip(event-8, 0, len(a)-32), np.clip(event-16, 0, len(a)-32)])
    vectors = []
    for t in starts:
        sl = slice(t,t+32)
        path = dp[sl].sum(0); angle = rot[sl].sum(0)
        displacement = np.linalg.norm(xyz[t+31]-xyz[t], axis=-1)
        turning = np.clip(1-displacement/np.maximum(path,1e-9), 0, 1)
        relative = abs(np.linalg.norm(xyz[t+31,0]-xyz[t+31,1])-np.linalg.norm(xyz[t,0]-xyz[t,1]))
        slow_angle = (rot[sl] * (dp[sl] <= np.quantile(dp, .5, axis=0))).sum(0)
        gap = np.linalg.norm((a[t,[0,1,2,8,9,10]]-s[t,[0,1,2,8,9,10]]).reshape(2,3), axis=-1)
        vectors.append(np.r_[path, angle, grip[sl].sum(0), displacement,
                             active[sl].mean(0), active[sl].all(-1).mean(), relative,
                             turning, slow_angle, a[sl][:,[7,15]].mean(0),
                             s[t,[0,1,2,8,9,10]], a[t,[7,15]], gap])
    return starts.astype(np.int64), np.asarray(vectors, dtype=np.float64)


def robust_transform(x):
    median = np.median(x, axis=0)
    scale = np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0)
    fallback = np.quantile(x,.95,axis=0)-np.quantile(x,.05,axis=0)
    scale = np.where(scale>1e-6, scale, np.where(fallback>1e-6, fallback, 1.))
    return median, scale


def transform(x, median, scale):
    return np.clip((x-median)/scale,-5,5)


def temporal_shortlist(starts, motion, limit=48):
    """Keep endpoints, broad phases, slow corrections, grips and both-arm cases."""
    if len(starts) <= limit:
        return np.arange(len(starts))
    chosen = set(np.rint(np.linspace(0,len(starts)-1,16)).astype(int).tolist())
    # Four candidates per event family, separated before this shortlist stage.
    for score in (motion[:,4:6].sum(-1), motion[:,14:16].sum(-1),
                  motion[:,10], motion[:,2:4].sum(-1), motion[:,12:14].sum(-1)):
        event_choices = []
        if np.ptp(score) <= 1e-12:
            continue
        for j in np.argsort(-score, kind='stable'):
            if all(abs(int(starts[j])-int(starts[k])) >= 16 for k in event_choices):
                chosen.add(int(j)); event_choices.append(int(j))
            if len(event_choices) == 4:
                break
    med, scale = robust_transform(motion)
    x = transform(motion,med,scale)
    while len(chosen) < min(limit,len(starts)):
        d = np.min(((x[:,None]-x[sorted(chosen)][None])**2).mean(-1),axis=1)
        d[list(chosen)] = -1
        chosen.add(int(np.argmax(d)))
    return np.array(sorted(chosen), dtype=np.int64)


def ordered_rgb(frames, starts):
    """Preserve the nine temporal slots. Sampling is selection-only, never policy RGB."""
    if frames.dtype != np.uint8 or tuple(frames.shape[1:]) != (3,240,416):
        raise ValueError('Unknown native RGB format')
    needed = sorted({f for t in starts for f in history_indices(t)})
    features = {}
    for f in needed:
        pixels = np.asarray(frames[f,:,::8,::8],dtype=np.float32)/255.
        features[f] = pixels.reshape(3,2,15,4,13).mean((2,4)).reshape(-1)
    return np.stack([np.concatenate([features[f] for f in history_indices(t)]) for t in starts])


def arrays_for(config, episode, cache_dir):
    base = Path(config['demonstrations'])/episode['path']
    expected = digest(dict(version=FEATURE_VERSION, episode=episode['id'], files=episode['files'],
                           stride=config['curation']['candidate_stride']))
    path = cache_dir/f"{int(episode['id']):05d}.npz"
    if path.exists():
        z = np.load(path,allow_pickle=False)
        if str(z['identity']) != expected:
            raise ValueError('Stale descriptor cache')
        return {k:z[k] for k in ('starts','motion','shortlist','visual')}
    # The RGB cache remains read-only; hashes are checked independently by verification.
    for key in ('actions','states'):
        if file_hash(base/f'{key}.npy') != episode['files'][f'{key}.npy']:
            raise ValueError(f'{key} source hash changed')
    a = np.load(base/'actions.npy',mmap_mode='r'); s = np.load(base/'states.npy',mmap_mode='r')
    starts,motion = motion_descriptors(a,s,config['curation']['candidate_stride'])
    shortlist = temporal_shortlist(starts,motion)
    frames = np.load(base/'frames.npy',mmap_mode='r')
    visual = ordered_rgb(frames,starts[shortlist])
    values = dict(starts=starts,motion=motion,shortlist=shortlist,visual=visual)
    with path.with_suffix('.partial').open('wb') as f:
        np.savez_compressed(f,identity=np.array(expected),**values)
    path.with_suffix('.partial').replace(path)
    return values


def split_episodes(rs, by_episode, seed, monitor_count):
    """Keep exact-content episode copies in the same group; do not count copies twice."""
    groups = defaultdict(list)
    for r in sorted(rs,key=lambda r:int(r['episode_index'])):
        e = by_episode[int(r['episode_index'])]
        identity = digest(dict(files=e['files'], prompt=e['prompt']))
        groups[identity].append(r)
    keys = sorted(groups)
    rng = np.random.default_rng(seed_for(seed,'curated-episode-split',rs[0]['task_name']))
    keys = [keys[i] for i in rng.permutation(len(keys))]
    if len(keys) <= monitor_count:
        raise ValueError('Insufficient independent episodes')
    monitors = [groups[k][0] for k in keys[:monitor_count]]
    fit = [groups[k][0] for k in keys[monitor_count:]]
    duplicates = [dict(episode=int(x['episode_index']),representative=int(groups[k][0]['episode_index']),
                       reason='identical_source_episode_hashes',group='monitor' if i<monitor_count else 'selection_pool')
                  for i,k in enumerate(keys) for x in groups[k][1:]]
    return fit, monitors, duplicates


def build_records(rows, by_episode, features, transform_stats, role):
    records=[]; matrices=[]
    med,scale=transform_stats
    for order,r in enumerate(rows):
        ep=int(r['episode_index']);e=by_episode[ep];z=features[ep]
        short=z['shortlist']; m=transform(z['motion'][short],med,scale)
        blocks=[m[:,idx]/np.sqrt(len(idx)) for idx in GROUPS.values()]
        # RGB retains colour and temporal identity; do not fit visual statistics on monitor.
        blocks.append(z['visual']/np.sqrt(z['visual'].shape[1]) * 2.)
        x=np.concatenate(blocks,axis=-1)/np.sqrt(len(blocks))
        for j,i in enumerate(short):
            t=int(z['starts'][i]);q=min(3,t*4//max(1,int(r['num_frames'])-31))
            proxy=[]
            if z['motion'][i,4:6].max()>.1:proxy.append('gripper_transition_proxy')
            if z['motion'][i,14:16].sum()>1e-5:proxy.append('slow_rotation_proxy')
            if z['motion'][i,10]>.1:proxy.append('both_arm_motion_proxy')
            suite='DLC_auxiliary' if r['dimension']=='additional' else r['dimension']
            protected=suite in ('memory','precision')
            records.append(dict(task=r['task_name'],suite=suite,episode=ep,frame=t,length=int(r['num_frames']),
                episode_path=e['path'],episode_order=order,temporal_quarter=q,role=role,
                text=e['text'],text_sha256=e['text_sha256'],prompt=e['prompt'],history_indices=history_indices(t),
                action_interval=[t,t+32],proxy_tags=proxy,protected_suite=protected,
                phase_status='unverified_proxy',memory_cue_observable=None if suite=='memory' else 'not_applicable',
                source_files=e['files'],descriptor_row=int(i),shortlist_row=j))
            matrices.append(x[j])
    return records,np.asarray(matrices)


def compatible(rows, selected, index):
    r=rows[index]
    return all(rows[j]['episode']!=r['episode'] or abs(rows[j]['frame']-r['frame'])>=32 for j in selected)


def capacity(rows, selected, allowed_episodes, cap):
    """Exact remaining interval capacity given already locked selections."""
    allowed=set(allowed_episodes);occupied=defaultdict(list);selected_ids=set(selected)
    for j in selected:occupied[rows[j]['episode']].append(rows[j]['frame'])
    eligible=defaultdict(list)
    for i,r in enumerate(rows):
        ep=r['episode']
        if ep in allowed and i not in selected_ids and all(abs(t-r['frame'])>=32 for t in occupied[ep]):
            eligible[ep].append(r['frame'])
    n=len(selected)
    for ep in allowed:
        last=-32;add=0
        for t in sorted(eligible[ep]):
            if t>=last+32:add+=1;last=t
        n+=min(cap-len(occupied[ep]),add)
    return n


def select_task(rows,x,target,min_episodes,cap,seed,random_order=False):
    """Protect temporal breadth, then reduce uncovered descriptor distance.

    Counts are not a guarantee of semantic phase coverage. That is reviewed later.
    """
    episode_ids=list(dict.fromkeys(r['episode'] for r in rows))
    by_ep={ep:[i for i,r in enumerate(rows) if r['episode']==ep] for ep in episode_ids}
    # Episode-level descriptor diversity, with deterministic ties.
    signatures=np.stack([np.r_[x[by_ep[ep]].mean(0),np.quantile(x[by_ep[ep]],[.1,.9],axis=0).ravel()] for ep in episode_ids])
    rng=np.random.default_rng(seed);chosen_eps=[int(rng.integers(len(episode_ids)))]
    distance=((signatures-signatures[chosen_eps[0]])**2).mean(-1)
    while len(chosen_eps)<len(episode_ids):
        distance[chosen_eps]=-1;j=int(np.argmax(distance));chosen_eps.append(j)
        distance=np.minimum(distance,((signatures-signatures[j])**2).mean(-1))
    order=episode_ids if random_order else [episode_ids[j] for j in chosen_eps]
    allowed=order[:min(min_episodes,len(order))]
    while capacity(rows,[],allowed,cap)<target and len(allowed)<len(order):allowed.append(order[len(allowed)])
    selected=[];reasons={};count=Counter();covered=defaultdict(set)
    # Initial anchors span time, but event proxies can replace ordinary anchors.
    for ep in allowed:
        candidates=by_ep[ep]
        for q in range(4):
            valid=[i for i in candidates if rows[i]['temporal_quarter']==q and compatible(rows,selected,i)]
            if not valid or len(selected)>=target:continue
            centre=(q+.5)/4
            # Prefer slow corrections/grips in Precision; broad native history in Memory.
            def score(i):
                r=rows[i];phase_bonus=.25*('slow_rotation_proxy' in r['proxy_tags'] or 'gripper_transition_proxy' in r['proxy_tags']) if r['suite']=='precision' else 0
                return phase_bonus-abs(r['frame']/max(1,r['length']-32)-centre)
            for i in sorted(valid,key=lambda j:(-score(j),rows[j]['frame'])):
                if capacity(rows,selected+[i],allowed,cap)>=min(target,capacity(rows,selected,allowed,cap)):
                    selected.append(i);reasons[i]='temporal_and_protected_proxy';count[ep]+=1;covered[ep].add(q);break
    if not selected:raise ValueError('No feasible selection')
    distances=np.min(((x[:,None]-x[selected][None])**2).sum(-1),axis=1)
    while len(selected)<target:
        valid=[i for ep in allowed if count[ep]<cap for i in by_ep[ep]
               if i not in reasons and compatible(rows,selected,i)]
        if not valid:
            if len(allowed)==len(order):break
            allowed.append(order[len(allowed)]);continue
        # Distance is novelty, not movement amplitude. Bound isolated outliers.
        scores=np.minimum(distances[valid],np.quantile(distances[valid],.95))
        for j,i in enumerate(valid):
            if rows[i]['temporal_quarter'] not in covered[rows[i]['episode']]:scores[j]*=1.15
        i=valid[int(np.argmax(scores))]
        selected.append(i);reasons[i]='additional_uncovered_pattern';count[rows[i]['episode']]+=1
        covered[rows[i]['episode']].add(rows[i]['temporal_quarter'])
        distances=np.minimum(distances,((x-x[i])**2).sum(-1))
    selected=sorted(selected,key=lambda i:(rows[i]['episode'],rows[i]['frame']))
    result=[dict(rows[i],selection_reason=reasons[i]) for i in selected]
    for ep in {r['episode'] for r in result}:
        starts=sorted(r['frame'] for r in result if r['episode']==ep)
        assert all(b-a>=32 for a,b in zip(starts,starts[1:]))
        assert len(starts)<=cap
    return result,selected,distances


def fingerprint_window(config,row):
    base=Path(config['demonstrations'])/row['episode_path']
    a=np.load(base/'actions.npy',mmap_mode='r');s=np.load(base/'states.npy',mmap_mode='r');f=np.load(base/'frames.npy',mmap_mode='r')
    h=hashlib.sha256();h.update(row['prompt'].encode());h.update(np.asarray(s[row['frame']]).tobytes())
    h.update(np.asarray(a[row['frame']:row['frame']+32]).tobytes())
    for t in row['history_indices']:h.update(np.asarray(f[t]).tobytes())
    return h.hexdigest()


def load_panel(path):
    p=json.loads(Path(path).read_text())
    if digest({k:v for k,v in p.items() if k!='identity'})!=p['identity']:
        raise ValueError('Panel identity mismatch')
    return p


def build(config):
    cfg=config['curation'];out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    cache=out/'descriptors';cache.mkdir(exist_ok=True)
    source=Path(config['demonstrations']);m=json.loads((source/'manifest.json').read_text())
    if digest({k:v for k,v in m.items() if k!='sha256'})!=m['sha256']:raise ValueError('Invalid source manifest identity')
    if m['preprocessing']!='native-tshape-bf16-roundtrip-anchor-history8-stride25-text-padding-v2':raise ValueError('Unknown preprocessing')
    by_episode={int(e['id']):e for e in m['episodes']}
    split=Path(config['reference_split']);train=list(csv.DictReader(open(split/'train.csv')));val=list(csv.DictReader(open(split/'val.csv')))
    if len(train)!=3466 or len(val)!=34:raise ValueError('Unexpected full dataset split')
    validation={int(r['episode_index']) for r in val};groups=defaultdict(list)
    for r in train:
        ep=int(r['episode_index']);e=by_episode[ep]
        if ep in validation or e['split']!='train' or e['task']!=r['task_name'] or e['length']!=int(r['num_frames']):raise ValueError('Split mismatch')
        if int(r['trim_start'])!=0 or int(r['trim_end'])!=e['length']:raise ValueError('Unsupported trimmed episode')
        groups[r['task_name']].append(r)
    if len(groups)!=35:raise ValueError('Expected 34 regular tasks and DLC')
    if json.loads((source/'dataset_stats.json').read_text())!=json.loads((Path(config['teacher_bundle'])/'dataset_stats.json').read_text()):raise ValueError('Normalization mismatch')
    selection=[];monitor=[];reserve=[];partitions=[];dedup=[];near=[];coverage={};features={};scalers={}
    for task_id,(task,rs) in enumerate(sorted(groups.items())):
        fit,held,duplicates=split_episodes(rs,by_episode,cfg['seed'],cfg['monitor_episodes_per_task']);dedup+=duplicates
        for n,r in enumerate(fit+held):
            ep=int(r['episode_index']);features[ep]=arrays_for(config,by_episode[ep],cache)
            if n%20==0:print(json.dumps(dict(event='descriptor_progress',task=task,episodes_done=n+1)),flush=True)
        motion=np.concatenate([features[int(r['episode_index'])]['motion'] for r in fit]);med,scale=robust_transform(motion)
        scalers[task]=dict(median=med.tolist(),scale=scale.tolist(),fit_episodes=[int(r['episode_index']) for r in fit])
        records,x=build_records(fit,by_episode,features,(med,scale),'selection_initial')
        sel,indices,distances=select_task(records,x,cfg['target_windows_per_task'],cfg['min_selection_episodes_per_task'],cfg['max_windows_per_episode'],seed_for(cfg['seed'],'select',task))
        # Exact-content deletions only. Near duplicates remain candidates for review.
        seen={};kept=[];removed=[]
        for row in sel:
            fp=fingerprint_window(config,row);row['content_sha256']=fp
            if fp in seen:
                old=seen[fp];dedup.append(dict(task=task,episode=row['episode'],frame=row['frame'],representative=[old['episode'],old['frame']],reason='identical_native_inputs_and_action_targets'));removed.append(row)
            else:seen[fp]=row;kept.append(row)
        sel=kept
        # Offer replacements only from the fit pool; never mine monitor for additions.
        for i in np.argsort(-distances,kind='stable'):
            if len(sel)>=cfg['target_windows_per_task']:break
            r=records[int(i)]
            if any(v['episode']==r['episode'] and abs(v['frame']-r['frame'])<32 for v in sel):continue
            if sum(v['episode']==r['episode'] for v in sel)>=cfg['max_windows_per_episode']:continue
            fp=fingerprint_window(config,r)
            if fp in seen:continue
            nr=dict(r,content_sha256=fp,selection_reason='nonredundant_replacement');sel.append(nr);seen[fp]=nr
        chosen_keys={(r['episode'],r['frame']) for r in sel};selected_eps={r['episode'] for r in sel}
        ordered=sorted(sel,key=lambda r:(r['episode'],r['frame']));ep_order={ep:i for i,ep in enumerate(sorted(selected_eps))}
        for r in ordered:r['episode_order']=ep_order[r['episode']];r['task_id']=task_id
        selection+=ordered
        held_rows,hx=build_records(held,by_episode,features,(med,scale),'monitor')
        mon,_,_=select_task(held_rows,hx,cfg['monitor_windows_per_task'],len(held),cfg['max_windows_per_episode'],seed_for(cfg['seed'],'monitor',task),True)
        for r in mon:
            r['audit_half']=r['episode_order']%2;r['task_id']=task_id
            r['content_sha256']=fingerprint_window(config,r)
        if {r['content_sha256'] for r in mon}&{r['content_sha256'] for r in sel}:raise ValueError('Exact content crosses selection and monitor')
        monitor+=mon
        for r in fit+held:
            ep=int(r['episode_index']);partitions.append(dict(task=task,episode=ep,role='monitor' if r in held else 'selection' if ep in selected_eps else 'reserve'))
        for r in records:
            if (r['episode'],r['frame']) not in chosen_keys:
                reserve.append(dict(r,task_id=task_id,role='reserve',shares_selected_episode=r['episode'] in selected_eps,
                                    eligible_for_automatic_expansion=False))
        # Suggest close pairs using a cheap descriptor. This cannot prove semantic redundancy.
        lookup={(r['episode'],r['frame']):i for i,r in enumerate(records)}
        si=[lookup[(r['episode'],r['frame'])] for r in sel]
        sx=x[si];d=((sx[:,None]-sx[None])**2).sum(-1);np.fill_diagonal(d,np.inf)
        for a in np.argsort(d.min(-1),kind='stable')[:10]:
            b=int(np.argmin(d[a]));ra,rb=sel[int(a)],sel[b]
            if (ra['episode'],ra['frame'])>(rb['episode'],rb['frame']):continue
            near.append(dict(task=task,first=[ra['episode'],ra['frame']],second=[rb['episode'],rb['frame']],
                             descriptor_distance=float(np.sqrt(d[a,b])),decision='retain_pending_review',
                             reason='descriptor_similarity_is_not_equivalence',protected=ra['protected_suite']))
        # Monitor descriptors evaluate breadth only, not policy performance.
        hd=np.min(((hx[:,None]-sx[None])**2).sum(-1),axis=1)**.5
        ep_errors=[hd[[i for i,r in enumerate(held_rows) if r['episode']==ep]].mean() for ep in sorted({r['episode'] for r in held_rows})]
        coverage[task]=dict(suite=ordered[0]['suite'],selected_windows=len(sel),selected_episodes=len(selected_eps),
            monitor_windows=len(mon),monitor_episodes=len({r['episode'] for r in mon}),
            windows_per_episode=dict(Counter(r['episode'] for r in sel)),
            temporal_quarters=dict(Counter(r['temporal_quarter'] for r in sel)),
            proxy_tags=dict(Counter(t for r in sel for t in r['proxy_tags'])),
            monitor_descriptor_distance=float(np.mean(ep_errors)),phase_verified=False,
            cue_observability_verified=False if ordered[0]['suite']=='memory' else 'not_applicable')
        print(json.dumps(dict(event='task_selected',task=task,**{k:coverage[task][k] for k in ('selected_windows','selected_episodes','monitor_windows')})),flush=True)
    all_rows=selection+monitor
    for i,r in enumerate(all_rows):
        r['id']=i;r['window_id']=digest([SCHEMA,m['sha256'],r['episode'],r['frame'],r['role']])
    for i,r in enumerate(reserve):
        r['id']=len(all_rows)+i;r['window_id']=digest([SCHEMA,m['sha256'],r['episode'],r['frame'],'reserve'])
    panel=dict(schema=SCHEMA,seed=cfg['seed'],source=str(source),source_identity=m['sha256'],source_manifest_sha256=file_hash(source/'manifest.json'),
        training_split_sha256=file_hash(split/'train.csv'),validation_split_sha256=file_hash(split/'val.csv'),
        normalization_sha256=file_hash(source/'dataset_stats.json'),curation_config=cfg,
        tasks=sorted(groups),windows=all_rows,counts=dict(Counter(r['role'] for r in all_rows)),episode_partitions=partitions,
        official_validation_episodes=sorted(validation),coverage=coverage,
        limitations=['Visual descriptors are ordered RGB tile proxies, not a learned semantic representation.',
                    'Phase, contact and memory-cue annotations require verification; proxies are not truth.',
                    'Ground-truth future RGB is not a policy input.',
                    'Training-only monitor is used for development; not final evaluation.'],
        open_suite='No training demonstrations; DLC remains separate.')
    panel['identity']=digest(panel)
    immutable(out/'panel.json',panel)
    immutable(out/'descriptor_scalers.json',scalers)
    for name,rows_ in [('selection_manifest.jsonl',selection),('monitor_manifest.jsonl',monitor),('reserve_manifest.jsonl',reserve),
                       ('redundancy_audit.jsonl',dedup),('near_duplicate_review.jsonl',near)]:
        content=''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows_)
        path=out/name
        if path.exists() and path.read_text()!=content:raise ValueError(f'Refusing to change immutable {path}')
        path.write_text(content)
    immutable(out/'episode_split.json',partitions)
    immutable(out/'coverage.json',dict(panel_identity=panel['identity'],by_task=coverage))
    immutable(out/'review_requirements.json',dict(panel_identity=panel['identity'],status='pending',
        required=['Review protected Precision/Memory phases and cue observability.',
                  'Review any proposed near-duplicate removal; uncertainty means retain.',
                  'Run native GPU input/future-invariance and mask-sensitivity checks.'],
        automatic_near_duplicate_deletions=0,exact_duplicates_removed=len(dedup)))
    return panel


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    config=json.loads(Path(a.config).read_text());panel=build(config)
    print(json.dumps(dict(event='curation_built',identity=panel['identity'],counts=panel['counts'])))


if __name__=='__main__':main()
