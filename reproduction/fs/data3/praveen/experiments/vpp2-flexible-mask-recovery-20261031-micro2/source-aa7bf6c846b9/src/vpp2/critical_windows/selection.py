"""Critical-first selection with explicit conflicts, never silent label dropping."""
from collections import Counter, defaultdict
import numpy as np
from vpp2.full_mask.curation_v3 import pack_capacity
from .contracts import critical_mask


class CoverageConflict(ValueError):
    pass



def critical_packing(frames, required, cap, preferred_starts=None):
    """Find a nonoverlapping packing that covers every required action step.

    Dynamic programming tracks the number of 32-action windows. A gap between
    consecutive windows is allowed only when it contains no required step.
    Maximize feasible packing capacity first, then minimize how many windows
    intersect critical phases. Retain only those critical windows; the full
    packing witnesses sufficient room for later coverage/random/diverse filler.
    """
    starts = np.asarray(frames, dtype=np.int64)
    if len(starts) == 0 or cap < 1:
        raise CoverageConflict('No candidate capacity for critical intervals')
    if not np.all(starts[1:] > starts[:-1]):
        raise ValueError('Episode candidates must have unique increasing frame starts')
    if starts.min() < 0 or starts.max() + 32 > len(required):
        raise ValueError('Candidate target interval exceeds episode bounds')
    prefix = np.r_[0, np.cumsum(required, dtype=np.int64)]
    ends = starts + 32
    before, after = prefix[starts], prefix[ends]
    hits = after > before
    costs = hits.astype(np.int32)
    if preferred_starts is not None:
        # Favor existing descriptor candidates when repairing a sparse grid.
        costs += (~np.isin(starts, preferred_starts)).astype(np.int32)*(cap+1)
    # Every predecessor ends before this window and leaves no uncovered critical
    # step in the intervening gap. Adjacent windows are allowed.
    predecessors = [np.flatnonzero((ends <= start) & (after == before[j]))
                    for j, start in enumerate(starts)]
    inf = cap + 1 if preferred_starts is None else cap*(cap+2)+1
    dp = np.full((cap+1, len(starts)), inf, dtype=np.int32)
    parent = np.full_like(dp, -1)
    dp[1, before == 0] = costs[before == 0]
    for count in range(2, cap+1):
        for j, candidates in enumerate(predecessors):
            if not len(candidates):
                continue
            i = int(candidates[np.argmin(dp[count-1, candidates])])
            if dp[count-1, i] < inf:
                dp[count, j] = dp[count-1, i] + costs[j]
                parent[count, j] = i
    complete = after == prefix[-1]
    for count in range(cap, 0, -1):
        candidates = np.flatnonzero(complete & (dp[count] < inf))
        if not len(candidates):
            continue
        j = int(candidates[np.argmin(dp[count, candidates])])
        path = []
        for k in range(count, 0, -1):
            path.append(j)
            j = int(parent[k, j])
        path.reverse()
        return [j for j in path if hits[j]], count
    raise CoverageConflict('Critical steps cannot all be covered by nonoverlapping candidates under the episode cap')



def repair_candidate_grid(rows, features, cap, annotations, observed_anchors):
    """Add only necessary native starts when the cached candidate grid has gaps.

    New starts are mandatory coverage windows, not proposals ranked by invented
    visual descriptors. Their actual native history/actions are read and hashed
    later. Diversity ranking continues to use only existing measured features.
    """
    from vpp2.full_mask.curation import history_indices
    rows=list(rows);features=np.asarray(features)
    by_episode=defaultdict(list)
    for i,row in enumerate(rows):by_episode[row['episode']].append(i)
    if (set(annotations)|set(observed_anchors))-set(by_episode):
        raise ValueError('Evidence episode absent from candidate grid')
    extras=[];plans={};repairs=[]
    for ep in sorted(set(annotations)|set(observed_anchors)):
        ids=by_episode[ep];template=rows[ids[0]];length=template['length']
        required=np.zeros(length,dtype=bool)
        if ep in annotations:
            a=annotations[ep]
            if a['role']!='fit' or a['status']!='verified':raise ValueError('Invalid fitting annotation')
            required|=critical_mask(0,length,length,a['intervals'])
        if ep in observed_anchors:
            a=observed_anchors[ep]
            if a['role']!='fit':raise ValueError('Only fitting anchors can repair candidates')
            if any(type(t) is not int or not 0<=t<length for t in a['frames']):raise ValueError('Bad anchor')
            required[a['frames']]=True
        if not required.any():continue
        starts=np.array(sorted(rows[i]['frame'] for i in ids))
        try:
            critical_packing(starts,required,cap)
            continue
        except CoverageConflict:
            all_starts=np.arange(length-31,dtype=np.int64)
            local,_=critical_packing(all_starts,required,cap,preferred_starts=starts)
        planned=[int(all_starts[i]) for i in local];plans[ep]=planned
        added=[];existing=set(starts.tolist())
        for t in planned:
            if t in existing:continue
            row={**template,'frame':t,'history_indices':history_indices(t),'action_interval':[t,t+32],
                 'temporal_quarter':min(3,t*4//max(1,length-31)),'relative_time':t/max(1,length-32),
                 'descriptor_row':None,'descriptor_available':False,'proxy_tags':[],
                 'candidate_origin':'required_coverage_repair','phase_status':'required_coverage_not_descriptor_ranked'}
            rows.append(row);extras.append(np.zeros(features.shape[1],dtype=features.dtype));added.append(t)
        repairs.append(dict(episode=ep,added_native_starts=added,required_packing=planned,
                            invented_visual_descriptors=False))
    if extras:features=np.concatenate([features,np.asarray(extras)],axis=0)
    return rows,features,plans,repairs


def choose(rows, features, target, seed, cap, annotations, base=4, random_fraction=.2, observed_anchors=None, preplanned_windows=None):
    """All annotated critical target steps are protected before ordinary coverage.

    This intentionally fails when candidate spacing, episode caps or total quota
    make that requirement impossible. The caller must report the affected phases
    instead of silently favoring a round target count. Monitoring annotations can
    never enter this selector.
    """
    if any(r['role'] != 'selection_initial' for r in rows):
        raise ValueError('Only fitting candidates may enter data selection')
    if any(a['role'] != 'fit' or a['status'] != 'verified' for a in annotations.values()):
        raise ValueError('Selection annotations must be verified fitting episodes')
    if not 0 <= random_fraction < 1:
        raise ValueError('Invalid random fraction')
    x=np.asarray(features);rng=np.random.default_rng(seed)
    if x.ndim!=2 or len(x)!=len(rows) or not np.isfinite(x).all():raise ValueError('Bad descriptors')
    episodes=sorted({r['episode'] for r in rows});indices={e:np.array(sorted((i for i,r in enumerate(rows) if r['episode']==e),key=lambda i:rows[i]['frame'])) for e in episodes}
    if set(annotations)-set(episodes):raise ValueError('Annotation/candidate task mismatch')
    frames=np.array([r['frame'] for r in rows]);eligible=np.ones(len(rows),dtype=bool)
    counts=Counter();chosen=[];reasons={};covered_quarters=defaultdict(set)
    remain={e:pack_capacity(frames[indices[e]],cap) for e in episodes}
    if sum(remain.values())<target:raise CoverageConflict('Insufficient nonoverlapping capacity')
    distances=np.full(len(rows),np.inf);required={}
    for e,a in annotations.items():
        required[e]=critical_mask(0,a['length'],a['length'],a['intervals'])
    # Coarse-screen anchors are verified visible frames, not interval labels.
    # Keep them separately named and reject monitoring information here too.
    for ep,a in (observed_anchors or {}).items():
        if ep not in indices or a.get('role')!='fit':
            raise ValueError('Visual anchors must belong to fitting candidates')
        length=rows[int(indices[ep][0])]['length']
        points=a.get('frames',[])
        if not points or any(type(t) is not int or not 0<=t<length for t in points):
            raise ValueError('Invalid observed manipulation anchor')
        if ep not in required:required[ep]=np.zeros(length,dtype=bool)
        required[ep][points]=True
    critical_plan={}
    for e,mask in required.items():
        if not mask.any():
            continue
        try:
            if e in (preplanned_windows or {}):
                starts=preplanned_windows[e]
                lookup={int(frames[i]):j for j,i in enumerate(indices[e])}
                if len(starts)!=len(set(starts)) or any(t not in lookup for t in starts):
                    raise ValueError('Invalid repair packing')
                local=[lookup[t] for t in sorted(starts)]
                covered=np.zeros_like(mask)
                for t in sorted(starts):
                    if covered[t:t+32].any():raise ValueError('Repair packing overlaps')
                    covered[t:t+32]=True
                if (mask & ~covered).any() or len(local)>cap:
                    raise CoverageConflict('Repair packing does not cover required frames within the cap')
                remaining=frames[indices[e]][[not covered[t:t+32].any() for t in frames[indices[e]]]]
                capacity=len(local)+pack_capacity(remaining,cap-len(local))
            else:
                local,capacity=critical_packing(frames[indices[e]],mask,cap)
        except CoverageConflict as exc:
            raise CoverageConflict(f'Episode {e}: critical frames cannot be covered: {exc}') from exc
        critical_plan[e]=indices[e][local]
        remain[e]=capacity
    if sum(remain.values())<target:
        raise CoverageConflict('Critical coverage leaves insufficient total nonoverlapping capacity')
    def can_add(i):
        e=rows[i]['episode'];ii=indices[e]
        mask=eligible[ii] & (np.abs(frames[ii]-frames[i])>=32)
        after=pack_capacity(frames[ii][mask],cap-counts[e]-1)
        return len(chosen)+1+sum(remain.values())-remain[e]+after>=target,mask,after
    def add(i,reason):
        if not eligible[i] or len(chosen)>=target:return False
        ok,mask,after=can_add(i)
        if not ok:return False
        e=rows[i]['episode'];chosen.append(int(i));reasons[int(i)]=reason;counts[e]+=1
        eligible[indices[e]]=mask if counts[e]<cap else False;remain[e]=after
        covered_quarters[e].add(rows[i]['temporal_quarter'])
        if rows[i].get('descriptor_available',True):
            np.minimum(distances,np.square(x-x[i]).sum(-1),out=distances)
        if e in required:required[e][frames[i]:frames[i]+32]=False
        return True
    # Use a whole-episode feasible packing, avoiding greedy overlap traps.
    for e,planned in sorted(critical_plan.items()):
        for i in planned:
            if not add(int(i),'verified_critical_interval' if e in annotations else 'observed_manipulation_anchor'):
                raise CoverageConflict(f'Episode {e}: global quota prevents its feasible critical packing')
        if required[e].any():
            raise AssertionError('Critical packing failed to cover its claimed steps')
    if any(eligible[i] and not row.get('descriptor_available',True) for i,row in enumerate(rows)):
        raise AssertionError('Unranked repair candidates must be selected only by the required packing')
    # Cover all episodes; critical windows already contribute toward this floor.
    for slot in range(base):
        for e in episodes:
            if counts[e]>slot:continue
            ids=indices[e][eligible[indices[e]]]
            order=sorted(ids,key=lambda i:(rows[i]['temporal_quarter']!=slot%4,
                         not bool(rows[i]['proxy_tags']),abs(rows[i]['relative_time']-(slot%4+.5)/4),frames[i]))
            for i in order:
                if add(int(i),'episode_coverage_and_event_candidate'):break
    random_goal=round(target*random_fraction)
    if len(chosen)+random_goal>target:
        raise CoverageConflict('Critical/episode coverage leaves insufficient room for the reserved random quota')
    random_done=0
    for i in rng.permutation(len(rows)):
        if random_done==random_goal:break
        if add(int(i),'balanced_random_coverage'):random_done+=1
    if random_done!=random_goal:raise CoverageConflict('Cannot preserve the specified random coverage')
    while len(chosen)<target:
        valid=np.flatnonzero(eligible)
        if not len(valid):raise CoverageConflict('No eligible filler candidates')
        novelty=distances[valid].copy();novelty=np.minimum(novelty,np.quantile(novelty,.95))
        score=novelty/max(float(novelty.max()),1e-12)
        score+=.02*np.array([rows[i].get('quality_rank',.5) if rows[i].get('quality_usable',False) else .5 for i in valid])
        score+=.03*np.array([rows[i]['temporal_quarter'] not in covered_quarters[rows[i]['episode']] for i in valid])
        for i in valid[np.argsort(-score,kind='stable')]:
            if add(int(i),'additional_distinct_behavior'):break
        else:raise CoverageConflict('No quota-feasible filler candidate')
    return sorted(chosen,key=lambda i:(rows[i]['episode'],rows[i]['frame'])),reasons
