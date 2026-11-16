import numpy as np
import pytest
from vpp2.full_mask.curation_v3 import allocate,pack_capacity,choose,dynamic_shortlist,relative_action_summary


def test_fair_capacity_allocation():
    assert allocate({'short':2,'a':10,'b':10},12)=={'a':5,'b':5,'short':2}
    with pytest.raises(ValueError):allocate({'a':1},2)


def test_variable_shortlist_keeps_all_packing_positions():
    starts=np.arange(0,1024-31,8);motion=np.random.default_rng(2).normal(size=(len(starts),28))
    short=dynamic_shortlist(starts,motion,1024)
    assert set(np.arange(0,1024-31,32))<=set(starts[short])
    assert pack_capacity(starts[short],16)==16


@pytest.mark.parametrize('target',[48,64])
def test_nonoverlap_feasible_with_broad_episode_coverage(target):
    rows=[]
    for ep in range(4):
        for t in range(0,512-31,8):
            rows.append(dict(episode=ep,frame=t,temporal_quarter=min(3,t*4//481),relative_time=t/480,
                             proxy_tags=['grip'] if t==104 else [],quality_rank=ep/4,quality_usable=True))
    x=np.random.default_rng(3).normal(size=(len(rows),12)).astype(np.float32)
    selected,reasons,d=choose(rows,x,target,100008,16)
    assert len(selected)==target
    assert set(rows[i]['episode'] for i in selected)==set(range(4))
    for ep in range(4):
        ts=sorted(rows[i]['frame'] for i in selected if rows[i]['episode']==ep)
        assert min(np.diff(ts))>=32
    assert choose(rows,x,target,100008,16)[0]==selected


def test_orientation_sign_invariance_and_rotation_direction():
    a=np.zeros((40,16));a[:,3]=1;a[:,11]=1;s=a.copy()
    a[:,3]=np.cos(.1);a[:,4]=np.sin(.1)
    f=relative_action_summary(a,s,0)
    b=a.copy();b[:,3:7]*=-1;b[:,11:15]*=-1
    ss=s.copy();ss[:,3:7]*=-1;ss[:,11:15]*=-1
    np.testing.assert_allclose(f,relative_action_summary(b,ss,0),atol=1e-6)
    b=a.copy();b[:,4]*=-1
    assert not np.allclose(f,relative_action_summary(b,s,0))


def test_observation_history_not_part_of_overlap_constraint():
    from vpp2.full_mask.curation import history_indices
    assert set(history_indices(64))&set(history_indices(96))
    assert pack_capacity([64,96])==2
