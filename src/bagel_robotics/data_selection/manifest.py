"""Content identities and deterministic task-balanced window sampling."""
from collections import defaultdict
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
