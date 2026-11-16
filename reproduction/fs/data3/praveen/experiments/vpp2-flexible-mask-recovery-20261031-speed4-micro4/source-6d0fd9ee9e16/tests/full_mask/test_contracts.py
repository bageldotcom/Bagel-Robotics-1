import copy
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from vpp2.full_mask.panel import choose_windows,BalancedStream
from vpp2.full_mask.controller import allocate_ffn,churn
from vpp2.learned_width_kv.reliability import action_losses,kv_losses


class Contracts(unittest.TestCase):
    def test_windows_temporal_coverage_and_sign_invariant_rotation(self):
        a=np.zeros((200,16),np.float32);a[:,3]=1;a[:,11]=1;a[100:,7]=1
        b=a.copy();b[::2,3]*=-1;b[::2,11]*=-1
        first=choose_windows(a,7,0);second=choose_windows(b,7,0)
        self.assertEqual(first,second)
        self.assertEqual({r['temporal_quarter'] for r in first},{0,1,2,3})
        self.assertEqual(len({r['frame'] for r in first}),4)
        self.assertTrue(all(0<=r['frame']<=168 for r in first))
        self.assertEqual(first,choose_windows(a,7,0))
    def test_partition_and_resume_independent_stream(self):
        rows=[dict(task=str(t),id=t*128+i) for t in range(35) for i in range(128)]
        a=BalancedStream(rows,100008)
        ids=[a.at(i)['id'] for i in range(4480)]
        self.assertEqual(len(set(ids)),4480)
        b=BalancedStream(rows,100008)
        self.assertEqual(ids[901:1234],[b.at(i)['id'] for i in range(901,1234)])
        ranks=[[a.at(i)['id'] for i in range(r,256,8)] for r in range(8)]
        self.assertEqual(sorted(sum(ranks,[])),sorted(ids[:256]))
        uneven=BalancedStream(rows[:-1],1)
        self.assertEqual(len({uneven.at(i)['task'] for i in range(35)}),35)
    def test_layer_specific_flexible_budget(self):
        scores=torch.tensor([[10.,9,8,7,6],[5.,4,3,2,1],[0.,-1,-2,-3,-4]])
        m=allocate_ffn(scores,8,1)
        self.assertEqual(int(m.sum()),8)
        self.assertEqual(m.sum(-1).tolist(),[5,2,1])
        self.assertEqual(int(m[2,0]),1)
        with self.assertRaises(ValueError):allocate_ffn(scores,2,1)
    def test_rejected_teacher_has_no_action_or_kv_gradient(self):
        p=torch.ones(1,32,16,requires_grad=True);teacher=torch.ones_like(p)*10;target=torch.zeros_like(p)
        table={'x':[dict(median=1.,q75=1.,q95=2.)]*4}
        gt,kd,_,w,_=action_losses(p,teacher,target,None,torch.ones(1),['x'],[0],table)
        cache=torch.ones(1,40,2,requires_grad=True);_,kv,_=kv_losses(cache,w)
        (gt+.1*kd+.05*kv).backward()
        self.assertGreater(float(p.grad.abs().sum()),0)
        self.assertEqual(float(cache.grad.abs().sum()),0)
    def test_churn_per_layer_not_suite_label(self):
        a=dict(hidden_indices=[0,1],ffn_indices=[[0,1],[0,1]])
        b=dict(hidden_indices=[0,2],ffn_indices=[[0,1],[1,2]])
        self.assertEqual(churn(a,b),dict(hidden=.5,ffn=[0.,.5]))
    def test_real_manifest_leakage_and_counts(self):
        path=Path('/data3/praveen/experiments/vpp2-full-robodojo-mask-calibration-20261028/data/curated-v2/panel.json')
        if not path.exists():self.fail(str(path))
        m=json.loads(path.read_text());parts={}
        for r in m['windows']:parts.setdefault(r['role'],set()).add(r['episode'])
        self.assertFalse(parts['monitor'] & parts['selection_initial'])
        self.assertFalse(set(m['official_validation_episodes']) & set.union(*parts.values()))
        self.assertEqual(m['counts'],dict(selection_initial=7000,monitor=2240))
        self.assertTrue(all(c['selected_episodes']>=25 for c in m['coverage'].values()))
        self.assertTrue(all(c['monitor_episodes']==16 for c in m['coverage'].values()))
        self.assertIn('insert_key',m['tasks']);self.assertIn('play_Xylophone',m['tasks'])

if __name__=='__main__':unittest.main()
