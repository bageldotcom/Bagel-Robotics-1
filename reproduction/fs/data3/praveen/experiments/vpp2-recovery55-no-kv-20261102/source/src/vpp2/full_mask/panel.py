"""Training-only episode partitions and reproducible, task-balanced windows.

Window priority uses demonstration commands, not simulator outcomes or validation
scores. Gripper/rotation scores are proxies, never contact or success labels.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(value):
    # JSON converts integer map keys to strings. Canonicalize before sorting so
    # identities survive save/load even for per-episode count dictionaries.
    canonical=json.loads(json.dumps(value,allow_nan=False))
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def seed_for(seed, *parts):
    return int(digest([seed, *parts])[:15], 16)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def immutable(path, value):
    path = Path(path)
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if path.exists() and path.read_text() != data:
        raise RuntimeError(f'Refusing to overwrite different artifact: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial'); temp.write_text(data); temp.replace(path)


def moving_max(values, horizon=24):
    # One score per complete 32-action window; look only within its executed prefix.
    return np.max(np.lib.stride_tricks.sliding_window_view(values, horizon), axis=-1)


def event_scores(actions):
    if actions.ndim != 2 or actions.shape[1] != 16 or not np.isfinite(actions).all():
        raise ValueError('Require finite native EE16 actions')
    delta = np.diff(actions, axis=0, prepend=actions[:1])
    gripper = np.max(np.abs(delta[:, [7, 15]]), axis=1)
    speed = np.linalg.norm(delta[:, [0,1,2,8,9,10]], axis=1)
    rotations = []
    for start in (3,11):
        q = actions[:, start:start+4].astype(np.float64)
        n = np.linalg.norm(q, axis=-1, keepdims=True)
        if np.any(n < 1e-8): raise ValueError('Invalid zero quaternion in demonstration')
        q = q/n
        rotations.append(2*np.arccos(np.clip(np.abs((q[1:]*q[:-1]).sum(-1)), 0, 1)))
    rotation = np.r_[0., np.maximum(*rotations)]
    # Within-episode ranks avoid mixing metres, quaternion coordinates and grip.
    def rank(x):
        if np.ptp(x) < 1e-9: return np.zeros_like(x)
        order = np.argsort(x, kind='stable')
        # Equal values receive the same rank; frame index cannot create saliency.
        unique, inverse = np.unique(x, return_inverse=True)
        return inverse/max(1, len(unique)-1)
    deceleration = np.maximum(0, -np.diff(speed, prepend=speed[0]))
    fine = np.maximum(rank(rotation), rank(deceleration)) * (1. - .5*rank(speed))
    return moving_max(gripper), moving_max(fine)


def choose_windows(actions, seed, order):
    """Four temporal quarters; two uniform, one grip-change, one fine-adjustment."""
    n = len(actions)-32+1
    if n < 4: raise ValueError('Episode too short for four unique full-action windows')
    grip, fine = event_scores(actions)
    rng = np.random.default_rng(seed)
    roles = ['uniform', 'uniform', 'gripper_change', 'rotation_or_deceleration']
    roles = np.roll(roles, order % 4).tolist()
    result = []
    for quarter, (indices, role) in enumerate(zip(np.array_split(np.arange(n), 4), roles)):
        actual = role
        if role == 'uniform':
            frame = int(rng.choice(indices)); score = 0.
        else:
            values = (grip if role == 'gripper_change' else fine)[indices]
            cutoff = np.quantile(values, .75)
            candidates = indices[(values >= cutoff) & (values > 1e-9)]
            if not len(candidates):
                candidates = indices; actual = 'uniform_fallback'
            frame = int(rng.choice(candidates)); score = float((grip if role == 'gripper_change' else fine)[frame])
        result.append(dict(frame=frame, temporal_quarter=quarter, selection=actual,
                           requested_selection=role, proxy_score=score))
    if len({x['frame'] for x in result}) != 4: raise AssertionError('Duplicate windows')
    return result


def build(config):
    source = Path(config['demonstrations'])
    manifest = json.loads((source/'manifest.json').read_text())
    if digest({k:v for k,v in manifest.items() if k!='sha256'}) != manifest['sha256']:
        raise ValueError('Decoded manifest identity changed')
    if manifest['preprocessing'] != 'native-tshape-bf16-roundtrip-anchor-history8-stride25-text-padding-v2':
        raise ValueError('Unknown preprocessing')
    split = Path(config['reference_split'])
    train = list(csv.DictReader(open(split/'train.csv')))
    val = list(csv.DictReader(open(split/'val.csv')))
    groups = defaultdict(list)
    for r in train: groups[r['task_name']].append(r)
    if len(groups) != 35 or len(train)!=3466 or len(val)!=34: raise ValueError('Full-scope split changed')
    validation = {int(r['episode_index']) for r in val}
    by_episode = {int(e['id']):e for e in manifest['episodes']}
    if len(by_episode)!=3500: raise ValueError('Expected all 3500 decoded episodes')
    normalizer = json.loads((source/'dataset_stats.json').read_text())
    if normalizer != json.loads((Path(config['teacher_bundle'])/'dataset_stats.json').read_text()):
        raise ValueError('Decoded data must use unchanged pretrained full-dataset normalization')
    rows=[]; partitions=[]; stats=defaultdict(lambda:dict(tasks=0,train_episodes=0,train_windows=0))
    for task_id, task in enumerate(sorted(groups)):
        candidates=sorted(groups[task], key=lambda r:int(r['episode_index']))
        if len(candidates)<80: raise ValueError('Need 64 selection + 16 audit episodes per task')
        rng=np.random.default_rng(seed_for(config['seed'], 'episode-partition', task))
        candidates=[candidates[i] for i in rng.permutation(len(candidates))]
        suite=candidates[0]['dimension']; suite='DLC_auxiliary' if suite=='additional' else suite
        stats[suite]['tasks']+=1; stats[suite]['train_episodes']+=len(candidates)
        stats[suite]['train_windows']+=sum(int(r['trim_end'])-int(r['trim_start']) for r in candidates)
        for order,r in enumerate(candidates):
            ep=int(r['episode_index']); e=by_episode[ep]
            if ep in validation or e['split']!='train' or e['task']!=task: raise ValueError('Split/task mismatch')
            if int(r['trim_start'])!=0 or int(r['trim_end'])!=e['length']: raise ValueError('Unexpected trim')
            role='selection_initial' if order<32 else 'selection_expansion' if order<64 else 'monitor' if order<80 else 'unused_reserve'
            partitions.append(dict(task=task,suite=suite,episode=ep,role=role,order=order))
            if role=='unused_reserve': continue
            action_path=source/e['path']/'actions.npy'
            if file_hash(action_path)!=e['files']['actions.npy']: raise ValueError('Action cache modified')
            actions=np.load(action_path,allow_pickle=False)
            windows=choose_windows(actions,seed_for(config['seed'],'window',ep),order)
            for w in windows:
                rows.append(dict(id=len(rows),task=task,task_id=task_id,suite=suite,episode=ep,
                    episode_path=e['path'],length=e['length'],role=role,episode_order=order,
                    audit_half=(order-64)%2 if role=='monitor' else order%2,
                    text=e['text'],text_sha256=e['text_sha256'],prompt=e['prompt'],**w))
    record=dict(schema='full-robodojo-calibration-v1',seed=config['seed'],source=str(source),
        source_manifest_sha256=file_hash(source/'manifest.json'),source_identity=manifest['sha256'],
        training_split_sha256=file_hash(split/'train.csv'),validation_split_sha256=file_hash(split/'val.csv'),
        normalization_sha256=file_hash(source/'dataset_stats.json'),suite_inventory=dict(stats),
        tasks=sorted(groups),counts=dict(Counter(r['role'] for r in rows)),
        official_validation_episodes=sorted(validation),episode_partitions=partitions,windows=rows,
        open_suite='No training demonstrations; evaluation only. DLC is not an Open training suite.',
        monitoring_scope='Training-only model-selection monitor, not an untouched final validation set.')
    record['identity']=digest(record)
    return record


class BalancedStream:
    """Random-access global stream; order independent of rank, batching and resume."""
    def __init__(self, rows, seed):
        self.groups=defaultdict(list); self.seed=seed; self.cache={}
        for r in rows:self.groups[r['task']].append(r)
        self.tasks=sorted(self.groups)
        if not self.tasks: raise ValueError('Nonempty window groups required')
        self.sizes={t:len(self.groups[t]) for t in self.tasks}
    def at(self, position):
        if position<0:raise ValueError('Negative sample cursor')
        round_,slot=divmod(position,len(self.tasks))
        # Randomize task order per round while preserving exactly one of each.
        rng=np.random.default_rng(seed_for(self.seed,'task-order',round_))
        task=self.tasks[int(rng.permutation(len(self.tasks))[slot])]
        cycle,offset=divmod(round_,self.sizes[task])
        key=(task,cycle)
        if key not in self.cache:
            self.cache[key]=np.random.default_rng(seed_for(self.seed,'window-order',task,cycle)).permutation(self.sizes[task])
            if len(self.cache)>len(self.tasks)*3:
                self.cache={k:v for k,v in self.cache.items() if k[1]>=cycle-1}
        row=self.groups[task][int(self.cache[key][offset])]
        return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    config=json.loads(Path(a.config).read_text());record=build(config)
    immutable(Path(config['root'])/'data/panel.json',record)
    summary={k:v for k,v in record.items() if k not in ('windows','episode_partitions')}
    immutable(Path(config['root'])/'checks/data_scope.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()


def noise_stratum(seed, position, task_count):
    """Randomized balanced strata independent of the selected action phase."""
    if position < 0 or task_count < 1:raise ValueError('Invalid position/task count')
    round_,slot=divmod(position,task_count)
    cycle,offset=divmod(round_,4)
    order=np.random.default_rng(seed_for(seed,'noise-strata-order',cycle,slot)).permutation(4)
    return int(order[offset])
