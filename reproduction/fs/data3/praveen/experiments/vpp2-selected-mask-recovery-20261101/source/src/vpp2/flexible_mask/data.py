"""Position-addressed, task-balanced sampling; sealed episodes cannot enter fitting."""
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from vpp2.full_mask.panel import BalancedStream, seed_for, digest


def validate_panel(panel):
    rows=panel['windows']
    fit=[r for r in rows if r['role']=='selection_initial']
    monitor=[r for r in rows if r['role']=='monitor']
    if len(fit)!=30000 or len(monitor)!=1119 or len(panel['tasks'])!=35:
        raise ValueError('Pinned panel counts changed')
    episodes={role:{p['episode'] for p in panel['episode_partitions'] if p['role']==role}
              for role in ('fit','monitor','audit')}
    if [len(episodes[k]) for k in episodes] != [2906,280,280]:
        raise ValueError('Episode partition changed')
    if len(set.union(*episodes.values())) != sum(map(len,episodes.values())):
        raise ValueError('Episode leakage')
    if set.union(*episodes.values()) & set(panel['official_validation_episodes']):
        raise ValueError('Official validation leakage')
    if len({r['id'] for r in rows})!=len(rows) or len({(r['episode'],r['frame']) for r in rows})!=len(rows):
        raise ValueError('Duplicate windows')
    for subset,role in ((fit,'fit'),(monitor,'monitor')):
        if {r['episode'] for r in subset} != episodes[role]:
            raise ValueError('Missing/forbidden episode in '+role)
        if {r['task'] for r in subset} != set(panel['tasks']):
            raise ValueError('Missing task')
        if any(not 0<=r['frame']<=r['length']-32 for r in subset):
            raise ValueError('Incomplete action chunk')
    return fit,monitor


def positions(update, rank, world, micro, global_batch=256):
    if global_batch % (world*micro): raise ValueError('Unequal accumulation')
    start=(update-1)*global_batch
    return [list(range(start+a*world*micro+rank*micro,start+a*world*micro+(rank+1)*micro))
            for a in range(global_batch//(world*micro))]


class EpisodeWindows:
    """All legal starts in fitting episodes, without millions of expanded JSON rows."""
    def __init__(self, prototypes):
        self.rows=sorted(prototypes,key=lambda r:r['episode'])
        self.ends=np.cumsum([r['length']-31 for r in self.rows]).tolist()

    def __len__(self): return self.ends[-1]

    def __getitem__(self,index):
        if not 0<=index<len(self): raise IndexError(index)
        ep=bisect_right(self.ends,index)
        frame=index-(self.ends[ep-1] if ep else 0)
        r=dict(self.rows[ep]);r['frame']=frame
        # Native episode/frame identity; disjoint from calibration cache keys.
        r['id']=1_000_000_000+r['episode']*1_000_000+frame
        if frame>=1_000_000: raise ValueError('Window identity range exceeded')
        r['history_indices']=[0]+[max(0,frame+o) for o in range(-175,1,25)]
        r['window_id']=digest([r['episode'],frame,'recovery-native-window-v1'])
        return r


def stream(panel,seed,recovery=False):
    fit,_=validate_panel(panel)
    result=BalancedStream(fit,seed)
    if recovery:
        by_episode={r['episode']:r for r in fit}
        groups=defaultdict(list)
        for r in by_episode.values():groups[r['task']].append(r)
        result.groups={t:EpisodeWindows(v) for t,v in groups.items()}
        result.sizes={t:len(v) for t,v in result.groups.items()}
    return result


def coverage(stream,cursor):
    # Every completed round visits each task exactly once. Partial rounds matter.
    rounds,tail=divmod(cursor,len(stream.tasks))
    counts={t:rounds for t in stream.tasks}
    if tail:
        for p in range(cursor-tail,cursor): counts[stream.at(p)['task']]+=1
    return {t:dict(visits=n,windows=stream.sizes[t],minimum_visits=n//stream.sizes[t],
                   cycles=n/stream.sizes[t],unique=min(n,stream.sizes[t])) for t,n in counts.items()}



def reader_configuration(config):
    """Reuse the parent's verified frozen-encoder cache across execution changes."""
    from .continuation import validate_parent
    current=config;visited=set()
    while current.get('resume_from'):
        if current['root'] in visited:raise ValueError('Cyclic execution continuation')
        visited.add(current['root'])
        parent,_=validate_parent(current,'mask')
        current=json.loads((parent/'config.json').read_text())
    return current


def _prepare_cached(model,reader,rows,positions,seed,strata):
    """Encode cache misses singly, then batch the deterministic prepared inputs.

    This keeps VAE/CLIP arithmetic identical to the verified microbatch-one path
    and handles partially cached batches. Noise is regenerated from each global
    position by native_prepare; neither noise nor teacher K/V are cached.
    """
    from vpp2.adaptive_mask.data import prepare as native_prepare
    if len(rows)!=len(positions) or len(rows)!=len(strata):
        raise ValueError('Window/position/noise-stratum length mismatch')
    cache=reader.runtime_cache
    if cache is not None and len(rows)>1:
        missing=[cache.load(row['id']) is None for row in rows]
        stats=getattr(reader,'batch_cache_statistics',dict(batches=0,mixed_batches=0,singly_encoded_windows=0))
        stats['batches']+=1;stats['mixed_batches']+=int(any(missing) and not all(missing))
        stats['singly_encoded_windows']+=sum(missing);reader.batch_cache_statistics=stats
        for row,position,stratum,absent in zip(rows,positions,strata,missing):
            if absent:
                native_prepare(model,reader,[row],[position],seed,[stratum])
                if cache.load(row['id']) is None:raise RuntimeError('Frozen conditioning cache miss was not populated')
    return native_prepare(model,reader,rows,positions,seed,strata)


class BatchCachedReads:
    """Reuse a verified payload within one batch; never skip its initial hash check."""
    def __init__(self, cache):self.cache=cache;self.payloads={}
    def __getattr__(self, name):return getattr(self.cache,name)
    def load(self,key):
        if key not in self.payloads:
            value=self.cache.load(key)
            if value is not None:self.payloads[key]=value
        return self.payloads.get(key)


def prepare_cached(model,reader,rows,positions,seed,strata):
    cache=reader.runtime_cache
    if cache is None or not getattr(reader,'reuse_prepared_payloads',False):
        return _prepare_cached(model,reader,rows,positions,seed,strata)
    try:
        reader.runtime_cache=BatchCachedReads(cache)
        return _prepare_cached(model,reader,rows,positions,seed,strata)
    finally:
        reader.runtime_cache=cache
