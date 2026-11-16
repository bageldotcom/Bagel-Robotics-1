"""Episode-broad curation. All labels from motion remain explicitly unverified.

Stages are resumable and separately identified. Final audit episodes are never
used to fit descriptors, quality models, choose samples or refresh masks.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from pathlib import Path
import numpy as np
from .panel import digest, file_hash, immutable, seed_for
from .curation import (history_indices, motion_descriptors, robust_transform,
                       transform, GROUPS, fingerprint_window, load_panel)

SCHEMA = 'robodojo-curated-calibration-v3'


def write_rows(path, rows):
    content=''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows)
    path=Path(path)
    if path.exists() and path.read_text()!=content:
        raise ValueError(f'Refusing to overwrite pinned artifact: {path}')
    temp=path.with_suffix('.partial');temp.write_text(content);temp.replace(path)


def pack_capacity(starts, cap=100000):
    last=-32;n=0
    for t in np.sort(starts):
        if t>=last+32:last=int(t);n+=1
    return min(n,cap)


def allocate(capacities, target):
    """Max-min fair integer quotas; saturate short tasks instead of duplicating."""
    if target>sum(capacities.values()):
        raise ValueError('Requested count exceeds true nonoverlapping capacity')
    quotas={t:0 for t in sorted(capacities)}
    while sum(quotas.values())<target:
        for t in quotas:
            if quotas[t]<capacities[t]:
                quotas[t]+=1
                if sum(quotas.values())==target:break
    return quotas


def inventory(config):
    out=Path(config['curation']['output']);out.mkdir(parents=True,exist_ok=True)
    src=Path(config['demonstrations']);m=json.loads((src/'manifest.json').read_text())
    if digest({k:v for k,v in m.items() if k!='sha256'})!=m['sha256']:
        raise ValueError('Invalid source manifest')
    old=load_panel(config['curation']['previous_panel'])
    if m['sha256']!=old['source_identity']:raise ValueError('Source changed since approved split')
    train=list(csv.DictReader(open(Path(config['reference_split'])/'train.csv')))
    val=list(csv.DictReader(open(Path(config['reference_split'])/'val.csv')))
    if len(train)!=3466 or len(val)!=34:raise ValueError('Unexpected source split')
    if file_hash(src/'dataset_stats.json')!=old['normalization_sha256']:
        raise ValueError('Native normalization changed')
    if json.loads((src/'dataset_stats.json').read_text())!=json.loads((Path(config['teacher_bundle'])/'dataset_stats.json').read_text()):
        raise ValueError('Native teacher normalizer mismatch')
    by={int(e['id']):e for e in m['episodes']};tasks=sorted({r['task_name'] for r in train})
    held={p['episode'] for p in old['episode_partitions'] if p['role']=='monitor'}
    final=set()
    for task in tasks:
        eps=sorted(int(r['episode_index']) for r in train if r['task_name']==task and int(r['episode_index']) in held)
        if len(eps)!=16:raise ValueError('Expected 16 held-out episodes per group')
        rng=np.random.default_rng(seed_for(config['seed'],'v3-audit-split',task))
        final.update(eps[i] for i in rng.permutation(16)[:8])
    records=[];seen={};capacities=Counter();issues=[]
    for r in train:
        ep=int(r['episode_index']);e=by[ep];role='audit' if ep in final else 'monitor' if ep in held else 'fit'
        if e['split']!='train' or e['task']!=r['task_name'] or e['length']!=int(r['num_frames']):
            raise ValueError('Episode/source split mismatch')
        ident=digest(dict(files=e['files'],prompt=e['prompt']))
        if ident in seen:raise ValueError('Duplicate episode requires explicit grouped split before selection')
        seen[ident]=ep
        base=src/e['path']
        for name in ('actions.npy','states.npy'):
            if file_hash(base/name)!=e['files'][name]:raise ValueError(f'Source changed: {base/name}')
        f=np.load(base/'frames.npy',mmap_mode='r',allow_pickle=False)
        if f.shape!=(e['length'],3,240,416) or f.dtype!=np.uint8:raise ValueError('Invalid native image geometry')
        if e['length']<32:raise ValueError('Episode lacks complete action window')
        suite='DLC_auxiliary' if r['dimension']=='additional' else r['dimension']
        capacity=min(config['curation']['max_windows_per_episode'],e['length']//32)
        records.append(dict(e,episode=ep,role=role,suite=suite,capacity=capacity,
                            success_status='not_independently_verified',task_id=tasks.index(e['task'])))
        if role=='fit':capacities[e['task']]+=capacity
    result=dict(schema=SCHEMA,seed=config['seed'],source=str(src),source_identity=m['sha256'],
        source_manifest_sha256=file_hash(src/'manifest.json'),normalization_sha256=old['normalization_sha256'],
        training_split_sha256=file_hash(Path(config['reference_split'])/'train.csv'),
        validation_split_sha256=file_hash(Path(config['reference_split'])/'val.csv'),
        official_validation_episodes=[int(r['episode_index']) for r in val],tasks=tasks,
        episodes=records,capacities=dict(capacities),quotas=allocate(capacities,config['curation']['target_total']),
        counts=dict(Counter(e['role'] for e in records)),source_issues=issues,
        initial_state_diversity='separate from episode quality',audit_use='sealed until final audit command')
    result['identity']=digest(result);immutable(out/'inventory.json',result)
    write_rows(out/'episode_split.jsonl',[dict(task=e['task'],episode=e['episode'],role=e['role']) for e in records])
    return result


def dynamic_shortlist(starts,motion,length):
    """Packing grid guarantees capacity; events and broad phases provide choices."""
    selected=set(np.searchsorted(starts,np.arange(0,length-31,32)).tolist())
    # Every packing-grid start exists because candidate stride divides 32.
    selected.update(np.rint(np.linspace(0,len(starts)-1,min(32,len(starts)))).astype(int).tolist())
    for col in ([4,5],[14,15],[10],[2,3],[12,13]):
        score=motion[:,col].sum(-1)
        if np.ptp(score)<1e-10:continue
        chosen=[]
        for i in np.argsort(-score,kind='stable'):
            if all(abs(int(starts[i])-int(starts[j]))>=16 for j in chosen):
                selected.add(int(i));chosen.append(int(i))
            if len(chosen)>=max(4,min(16,length//128)):break
    return np.array(sorted(selected),dtype=np.int64)


def relative_action_summary(a,s,t):
    """Selection-only physical summaries; no change to policy action labels."""
    chunk=np.asarray(a[t:t+32],dtype=np.float64);state=np.asarray(s[t],dtype=np.float64)
    result=[]
    for off in (0,8):
        p=chunk[:,off:off+3]-state[off:off+3]
        q=chunk[:,off+3:off+7];q=q/np.linalg.norm(q,axis=-1,keepdims=True)
        ref=state[off+3:off+7];ref=ref/np.linalg.norm(ref)
        ref=ref*(1 if ref[np.argmax(np.abs(ref))]>=0 else -1)
        q=q*np.where((q@ref)>=0,1.,-1.)[:,None]
        # Sign-invariant geodesic change from the current rotation.
        angle=2*np.arccos(np.clip(np.abs(q@ref),0,1))
        grip=chunk[:,off+7]
        result.append(np.c_[p,q-ref,angle,grip].reshape(8,4,9).mean(1).ravel())
    return np.concatenate(result).astype(np.float32)


def episode_candidates(args):
    config,e=args;out=Path(config['curation']['output'])/'candidates';out.mkdir(exist_ok=True)
    path=out/f"{e['episode']:05d}.npz"
    ident=digest(dict(episode=e,schema=SCHEMA,stride=config['curation']['candidate_stride']))
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            if str(z['identity'])!=ident:raise ValueError('Stale candidate cache')
            return dict(episode=e['episode'],candidates=len(z['starts']))
    base=Path(config['demonstrations'])/e['path']
    a=np.load(base/'actions.npy',mmap_mode='r');s=np.load(base/'states.npy',mmap_mode='r')
    old=Path(config['curation']['previous_panel']).parent/'descriptors'/f"{e['episode']:05d}.npz"
    expected_old=digest(dict(version='physical-motion-and-ordered-rgb-tiles-v2',episode=e['id'],files=e['files'],stride=config['curation']['candidate_stride']))
    if old.exists():
        with np.load(old,allow_pickle=False) as z:
            if str(z['identity'])!=expected_old:raise ValueError('Old descriptor identity mismatch')
            starts=z['starts'];motion=z['motion']
    else:
        starts,motion=motion_descriptors(a,s,config['curation']['candidate_stride'])
    short=dynamic_shortlist(starts,motion,e['length'])
    chosen=starts[short]
    action=np.stack([relative_action_summary(a,s,int(t)) for t in chosen])
    # Uniform-in-time scoring pool, separate from event-enriched selection.
    score_starts=np.arange(0,e['length']-31,32,dtype=np.int64)
    score_indices=np.searchsorted(chosen,score_starts)
    if not np.array_equal(chosen[score_indices],score_starts):raise AssertionError('Missing uniform scoring pool')
    needed=np.array(sorted({f for t in chosen for f in history_indices(t)}),dtype=np.int64)
    temp=path.with_suffix('.partial')
    with temp.open('wb') as f:np.savez_compressed(f,identity=np.array(ident),starts=chosen,
        motion=motion[short].astype(np.float32),action=action,state=np.asarray(s[chosen],dtype=np.float32),score_indices=score_indices,needed_frames=needed)
    temp.replace(path)
    return dict(episode=e['episode'],candidates=len(chosen),frames=len(needed))


def candidates(config,workers):
    inv=json.loads((Path(config['curation']['output'])/'inventory.json').read_text())
    # Final audit observations are intentionally not summarized or scored here.
    todo=[e for e in inv['episodes'] if e['role']!='audit']
    result=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,r in enumerate(pool.map(episode_candidates,[(config,e) for e in todo]),1):
            result.append(r)
            if i%100==0:print(json.dumps(dict(event='candidates',done=i,total=len(todo))),flush=True)
    immutable(Path(config['curation']['output'])/'candidate_receipt.json',dict(inventory_identity=inv['identity'],episodes=result))


def minimum_distances(x, selected):
    d=np.full(len(x),np.inf,dtype=np.float32)
    for i in selected:d=np.minimum(d,np.square(x-x[i]).sum(-1))
    return d


def choose(rows,x,target,seed,cap,base=4,random_fraction=.2):
    """Feasible selection with episode floors, bounded novelty and quality tie-breaks."""
    rng=np.random.default_rng(seed);eps=sorted({r['episode'] for r in rows})
    idx={e:np.array([i for i,r in enumerate(rows) if r['episode']==e]) for e in eps}
    frames=np.array([r['frame'] for r in rows]);eligible=np.ones(len(rows),dtype=bool)
    counts=Counter();selected=[];reasons={};covered=defaultdict(set)
    remain={e:pack_capacity(frames[idx[e]],cap) for e in eps}
    if sum(remain.values())<target:raise ValueError('Insufficient actual candidate capacity')
    dist=np.full(len(rows),np.inf,dtype=np.float32)
    def add(i,reason):
        e=rows[i]['episode'];ii=idx[e]
        mask=eligible[ii] & (np.abs(frames[ii]-frames[i])>=32)
        after=pack_capacity(frames[ii][mask],cap-counts[e]-1)
        if len(selected)+1+sum(remain.values())-remain[e]+after<target:return False
        selected.append(int(i));reasons[int(i)]=reason;counts[e]+=1
        eligible[ii]=mask if counts[e]<cap else False
        remain[e]=after;covered[e].add(rows[i]['temporal_quarter'])
        np.minimum(dist,np.square(x-x[i]).sum(-1),out=dist)
        return True
    # Cover every episode before exploiting additional patterns in long episodes.
    for slot in range(base):
        for e in eps:
            if len(selected)>=target:break
            ids=idx[e][eligible[idx[e]]]
            if not len(ids):continue
            q=slot%4
            order=sorted(ids,key=lambda i:(rows[i]['temporal_quarter']!=q,
                not bool(rows[i]['proxy_tags']),abs(rows[i]['relative_time']-(q+.5)/4),frames[i]))
            for i in order:
                if add(i,'episode_coverage_and_event_candidate'):break
    # A fixed random budget avoids letting the learned features define all coverage.
    random_goal=min(round(target*random_fraction),target-len(selected));random_done=0
    for i in rng.permutation(len(rows)):
        if random_done>=random_goal:break
        if eligible[i] and add(int(i),'balanced_random_coverage'):random_done+=1
    while len(selected)<target:
        valid=np.flatnonzero(eligible)
        if not len(valid):raise AssertionError('Feasibility invariant broken')
        novelty=dist[valid].copy();cut=np.quantile(novelty,.95);novelty=np.minimum(novelty,cut)
        scale=max(float(np.max(novelty)),1e-12)
        # Quality may break near ties, never replace the coverage requirement.
        score=novelty/scale
        score+=.02*np.array([rows[i].get('quality_rank',.5) if rows[i].get('quality_usable',False) else .5 for i in valid])
        score+=.03*np.array([rows[i]['temporal_quarter'] not in covered[rows[i]['episode']] for i in valid])
        accepted=False
        for i in valid[np.argsort(-score,kind='stable')]:
            if add(int(i),'additional_distinct_behavior'):accepted=True;break
        if not accepted:raise AssertionError('No feasible candidate despite capacity')
    return sorted(selected,key=lambda i:(rows[i]['episode'],rows[i]['frame'])),reasons,dist


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--stage',choices=['inventory','candidates','select'],required=True);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();c=json.loads(Path(a.config).read_text())
    if a.stage=='inventory':r=inventory(c);print(json.dumps({k:r[k] for k in ('counts','quotas','capacities','identity')}))
    elif a.stage=='candidates':candidates(c,a.workers)
    else:
        from .curation_v3_finalize import build
        build(c)

if __name__=='__main__':main()
