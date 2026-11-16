import numpy as np
import pytest
from vpp2.critical_windows.selection import choose,CoverageConflict


def candidates():
    rows=[]
    for e in (1,2):
        for t in range(0,193,8):
            rows.append(dict(episode=e,frame=t,length=224,role='selection_initial',
                        temporal_quarter=min(3,t//56),relative_time=t/192,
                        proxy_tags=[],quality_rank=.5,quality_usable=False))
    return rows,np.random.default_rng(9).normal(size=(len(rows),4))


def annotation(start=112,end=128):
    return {1:dict(role='fit',status='verified',length=224,
                   intervals=[dict(frames=[start,end],kind='precision_alignment')])}


def test_critical_before_coverage_and_random_no_overlap():
    rows,x=candidates();chosen,reasons=choose(rows,x,10,7,7,annotation(),base=2)
    selected=[rows[i] for i in chosen]
    assert len(chosen)==10
    assert list(reasons.values()).count('balanced_random_coverage')==2
    assert 'verified_critical_interval' in reasons.values()
    assert any(r['episode']==1 and r['frame']<=112 and r['frame']+32>=128 for r in selected)
    for ep in (1,2):
        starts=sorted(r['frame'] for r in selected if r['episode']==ep)
        assert all(b-a>=32 for a,b in zip(starts,starts[1:]))
    assert choose(rows,x,10,7,7,annotation(),base=2)[0]==chosen


def test_monitor_annotations_rejected():
    rows,x=candidates();a=annotation();a[1]['role']='monitor'
    with pytest.raises(ValueError,match='fitting'):choose(rows,x,10,7,7,a,base=2)


def test_critical_tail_absent_from_candidates_is_a_reported_conflict():
    rows,x=candidates();a=annotation(220,224)
    restricted=[i for i,r in enumerate(rows) if r['frame']<192]
    with pytest.raises(CoverageConflict,match='critical frames'):
        choose([rows[i] for i in restricted],x[restricted],8,7,6,a,base=2)
