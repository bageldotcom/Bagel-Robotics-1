from collections import Counter
import copy
import json
from pathlib import Path
import random
import numpy as np
import pytest
import torch
from vpp2.adaptive_mask.state import snapshot,restore,rng_state,diagnostic_rng,save
from vpp2.adaptive_mask.controller import Controller
from vpp2.adaptive_mask.geometry import Geometry
from vpp2.adaptive_mask.reliability import weights
from vpp2.adaptive_mask.data import future_mask
from vpp2.adaptive_mask.losses import objectives,video_error
from vpp2.adaptive_mask.quality import stable,assess
from vpp2.full_mask.panel import BalancedStream


def controller():return Controller(Geometry(hidden=4,ffn=8,layers=3,heads=2,head_dim=2,pack=2,hidden_floor=2,ffn_floor=2,head_floor=1))

def test_restart_and_rollback_restore_actual_optimizer_rng_and_cursor(tmp_path):
    c=controller();o=torch.optim.AdamW(c.parameters(),lr=.01,weight_decay=0.)
    def update():
        o.zero_grad(set_to_none=True)
        loss=sum((g*torch.randn_like(g)).sum() for g in c.gates());loss.backward();o.step()
        return float(loss.detach()),copy.deepcopy(c.state_dict()),copy.deepcopy(o.state_dict())
    update();s=snapshot(c,o,256,1,Counter({5:2,9:1}),0)
    save(tmp_path/'checkpoint.pt',s);expected=update()
    actual_state=torch.load(tmp_path/'checkpoint.pt',weights_only=False)
    cursor,update_id,coverage,ramp=restore(actual_state,c,o);actual=update()
    assert cursor==256 and update_id==1 and coverage==Counter({5:2,9:1}) and ramp==0
    assert actual[0]==expected[0]
    for k in ('hidden_scores','ffn_scores','cross_scores'):assert torch.equal(actual[1][k],expected[1][k])
    for key in expected[2]['state']:
        for name,value in expected[2]['state'][key].items():assert torch.equal(value,actual[2]['state'][key][name])


def test_diagnostics_do_not_advance_rng():
    torch.manual_seed(3);s=rng_state()
    with diagnostic_rng():torch.rand(20);np.random.rand(20);random.random()
    a=torch.rand(10)
    from vpp2.adaptive_mask.state import restore_rng
    restore_rng(s);assert torch.equal(a,torch.rand(10))


def test_real_panel_rank_independent_sampling():
    panel=json.loads(Path('/data3/praveen/experiments/vpp2-critical-calibration-20261029/revisions/3ca392bfb574898f/panel.json').read_text())
    fitting=[r for r in panel['windows'] if r['role']=='selection_initial'];stream=BalancedStream(fitting,100008)
    serial=[stream.at(i)['id'] for i in range(1024)]
    for world in [1,2,8]:
        rank_sequences=[[stream.at(i)['id'] for i in range(rank,1024,world)] for rank in range(world)]
        assert [rank_sequences[i%world][i//world] for i in range(1024)]==serial
    resumed=BalancedStream(fitting,100008)
    assert [resumed.at(i)['id'] for i in range(511,1024)]==serial[511:]


def test_error_ratio_has_scale_invariance_and_component_veto():
    t=torch.tensor([.1,.5]);s=torch.tensor([.1,1.]);z=torch.ones(2)
    w,_=weights(t,s,z)
    for scale in [1e-4,1.,1e4]:
        q,_=weights(t*scale,s*scale,z*scale)
        torch.testing.assert_close(w,q)
    w,why=weights(t,s,z,torch.tensor([[.1,.1],[.1,2.]]),torch.ones(2,2))
    torch.testing.assert_close(w,torch.tensor([.9,0.]));assert why['component_veto'].tolist()==[False,True]
    w,_=weights(torch.tensor([0.,1e-4]),torch.ones(2),torch.zeros(2))
    assert w.tolist()==[1.,0.]


def test_future_complete_horizon_and_no_zero_denominator():
    rows=[dict(frame=0,length=n) for n in (16,17,32,33)]
    assert future_mask(rows,'cpu').tolist()==[[0,0],[1,0],[1,0],[1,1]]
    a=torch.ones(4,2,5,2,2,requires_grad=True);b=torch.zeros_like(a)
    loss,valid=video_error(a,b,future_mask(rows,'cpu'));assert loss.tolist()==[0,1,1,1]
    loss.sum().backward();assert a.grad[0].count_nonzero()==0 and a.grad[:,:,:3].count_nonzero()==0


def test_microbatch_partition_leaves_gated_losses_and_gradients_unchanged():
    results=[]
    for micro in (1,2,4):
        parameter=torch.nn.Parameter(torch.tensor(.5));total=0.
        for ids in torch.arange(4).split(micro):
            n=len(ids);target=torch.full((n,3,2),4.)
            student=parameter.expand_as(target);teacher=torch.stack([torch.full((3,2),3.5 if int(i)%2 else 0.) for i in ids])
            video=parameter.expand(n,1,5,2,2);kv=parameter.expand(n,40,2)
            loss,metrics,_,_=objectives(student,teacher,target,None,torch.ones(n),video,torch.zeros_like(video),kv,torch.ones(n,2),100)
            (loss*n/4).backward();total+=float(loss.detach())*n/4
        results.append((total,parameter.grad.item()))
    for value in results[1:]:np.testing.assert_allclose(value,results[0],rtol=1e-6)


def test_final_stability_checks_cumulative_drift_and_active_gradients():
    selection=controller().selection();rs=[dict(lineage_update=i*70,cap=100,selection=copy.deepcopy(selection),quality_passed=True,gradients_active=True) for i in range(5)]
    assert stable(rs)
    rs[-1]['gradients_active']=False;assert not stable(rs)
    rs[-1]['gradients_active']=True;rs[-1]['selection']['cross_head_indices'][0]=[1];assert not stable(rs)
