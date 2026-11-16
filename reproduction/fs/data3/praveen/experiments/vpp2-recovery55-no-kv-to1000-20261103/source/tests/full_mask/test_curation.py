import unittest
from collections import Counter
import numpy as np

from vpp2.full_mask.curation import (candidate_starts,history_indices,motion_descriptors,
    select_task,capacity,split_episodes,ordered_rgb)
from vpp2.full_mask.panel import BalancedStream,noise_stratum


def actions(n):
    a=np.zeros((n,16),np.float64);a[:,3]=1;a[:,11]=1
    return a


def fake_rows(episodes=25,length=201):
    rows=[]
    for ep in range(episodes):
        for t in candidate_starts(length):
            rows.append(dict(episode=ep,frame=int(t),length=length,task='test',suite='precision',
                temporal_quarter=min(3,int(t)*4//max(1,length-31)),proxy_tags=[]))
    x=np.array([[r['frame']/length,r['episode']/max(1,episodes)] for r in rows])
    return rows,x


class Curation(unittest.TestCase):
    def test_inclusive_last_full_action_window(self):
        self.assertEqual(candidate_starts(32).tolist(),[0])
        self.assertEqual(candidate_starts(31).tolist(),[])
        self.assertEqual(candidate_starts(105)[-1],73)
        self.assertEqual(history_indices(50),[0,0,0,0,0,0,0,25,50])
    def test_rotation_sign_invariance_and_fine_motion(self):
        a=actions(100);a[:,0]=np.linspace(0,.001,100);a[50:,7]=1
        b=a.copy();b[::2,3:7]*=-1;b[::3,11:15]*=-1
        ts,x=motion_descriptors(a,a);us,y=motion_descriptors(b,b)
        np.testing.assert_array_equal(ts,us);np.testing.assert_allclose(x,y,atol=1e-12)
        self.assertGreater(x[:,0].max(),0)
        self.assertLess(x[:,0].max(),.001)
    def test_zero_quaternion_rejected(self):
        a=actions(40);a[2,3:7]=0
        with self.assertRaises(ValueError):motion_descriptors(a,a)
    def test_nonoverlap_with_short_episodes_requires_expansion(self):
        rows,x=fake_rows(45,177)
        chosen,ids,_=select_task(rows,x,200,25,12,100008)
        self.assertEqual(len(chosen),200)
        self.assertGreaterEqual(len({r['episode'] for r in chosen}),40)
        for ep in {r['episode'] for r in chosen}:
            ts=sorted(r['frame'] for r in chosen if r['episode']==ep)
            self.assertTrue(all(b-a>=32 for a,b in zip(ts,ts[1:])))
        repeat=select_task(rows,x,200,25,12,100008)[1]
        self.assertEqual(ids,repeat)
    def test_capacity_handles_locked_intervals(self):
        rows,x=fake_rows(1,100)
        self.assertEqual(capacity(rows,[],[0],12),3)
        i=next(i for i,r in enumerate(rows) if r['frame']==24)
        self.assertEqual(capacity(rows,[i],[0],12),2)
    def test_temporal_visual_order_is_retained(self):
        f=np.zeros((200,3,240,416),np.uint8)
        f[0]=255;f[25,0]=255
        x=ordered_rgb(f,[25,50]);self.assertEqual(x.shape,(2,216))
        self.assertFalse(np.array_equal(x[0],x[1]))
    def test_unequal_task_counts_and_resume(self):
        rows=[dict(task='a',id=i) for i in range(3)]+[dict(task='b',id=10+i) for i in range(7)]
        stream=BalancedStream(rows,100008);items=[stream.at(i) for i in range(140)]
        self.assertEqual(Counter(x['task'] for x in items),{'a':70,'b':70})
        for task,n in [('a',3),('b',7)]:
            ids=[r['id'] for r in items if r['task']==task]
            for k in range(len(ids)//n):self.assertEqual(len(set(ids[k*n:(k+1)*n])),n)
        resumed=BalancedStream(rows,100008)
        self.assertEqual(items[17:80],[resumed.at(i) for i in range(17,80)])
        self.assertEqual(sorted(x['id'] for x in items[:64]),
            sorted(stream.at(i)['id'] for r in range(8) for i in range(r,64,8)))
    def test_noise_draws_are_rank_independent_and_stratified(self):
        expected=[noise_stratum(100008,i,35) for i in range(140)]
        self.assertEqual(Counter(expected),{0:35,1:35,2:35,3:35})
        self.assertEqual(expected[81:109],[noise_stratum(100008,i,35) for i in range(81,109)])
    def test_duplicate_episodes_cannot_cross_partition(self):
        rs=[];meta={}
        for i in range(10):
            rs.append(dict(episode_index=str(i),task_name='x'))
            meta[i]=dict(id=str(i),files={'frames.npy':str(i//2)},prompt='same')
        fit,mon,dupes=split_episodes(rs,meta,100008,2)
        self.assertEqual(len(fit),3);self.assertEqual(len(mon),2);self.assertEqual(len(dupes),5)
        self.assertFalse({int(r['episode_index'])//2 for r in fit}&{int(r['episode_index'])//2 for r in mon})


if __name__=='__main__':unittest.main()
