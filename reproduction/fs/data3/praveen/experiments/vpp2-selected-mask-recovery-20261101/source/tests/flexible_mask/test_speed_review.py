import json
import pytest
import torch
from vpp2.flexible_mask.config import identity
from vpp2.flexible_mask.speed_review import compare


def evidence(tmp_path,variant='micro4',speedup=1.25,delta=.015):
    c={'root':str(tmp_path),'resume_from':{'identity':'parent'}};ident=identity(c)
    (tmp_path/'checks').mkdir()
    for name in ('baseline',variant):
        p=tmp_path/'speed_review'/name;p.mkdir(parents=True)
        factor=1. if name=='baseline' else 1+delta
        rows=[];predictions=[];gradients=[]
        for u in range(3):
            rows.append(dict(update=u+10,seconds_per_update=100 if name=='baseline' else 100/speedup,
                samples_sha256=str(u),gt_action=factor,teacher_video_gated=factor,late_kv_gated=factor,
                total=factor,video_gradient_norm=factor))
            predictions.append([(i,torch.ones(2,2)*factor,torch.ones(2,2)*factor,torch.tensor(.5)) for i in range(4)])
            gradients.append([torch.ones(4)*factor for _ in range(3)])
        (p/'metrics.json').write_text(json.dumps(rows))
        torch.save(dict(predictions=predictions,mask_gradients=gradients),p/'comparison_tensors.pt')
        for rank in range(8):
            item=dict(identity=ident,start=9,end=12,initial={'state':'same'},final={'state':'same' if delta==0 or name=='baseline' else 'different'},
                      frozen_unchanged=True,calibration={'gt':[1,1,1]},max_allocated_gib=60,
                      micro_batch=2 if name=='baseline' or variant!='micro4' else 4,execution={})
            (p/f'rank{rank}.json').write_text(json.dumps(item))
    return c


def test_relaxed_rounding_requires_substantial_speedup(tmp_path):
    c=evidence(tmp_path)
    r=compare(c,'micro4')
    assert r['passed'] and r['relaxed_bf16_tolerance_used'] and r['output_tolerance']==.02


def test_small_speedup_cannot_use_relaxed_tolerance(tmp_path):
    c=evidence(tmp_path,speedup=1.1)
    r=compare(c,'micro4')
    assert not r['passed'] and not r['relaxed_bf16_tolerance_used']


def test_same_batch_must_preserve_state_exactly(tmp_path):
    c=evidence(tmp_path,variant='optimized',delta=.00001)
    assert 'same-microbatch-state-not-exact' in compare(c,'optimized')['failures']


@pytest.mark.parametrize('problem',['initial','frozen','memory','gate'])
def test_correctness_checks_are_not_relaxed(tmp_path,problem):
    c=evidence(tmp_path,delta=0)
    p=tmp_path/'speed_review/micro4/rank0.json';item=json.loads(p.read_text())
    if problem=='initial':item['initial']={'state':'changed'}
    if problem=='frozen':item['frozen_unchanged']=False
    if problem=='memory':item['max_allocated_gib']=77
    if problem=='gate':
        p2=tmp_path/'speed_review/micro4/comparison_tensors.pt'
        tensors=torch.load(p2,weights_only=True)
        p0,t,a,r=tensors['predictions'][0][0]
        tensors['predictions'][0][0]=(p0,t,a,torch.tensor(0.))
        torch.save(tensors,p2)
    p.write_text(json.dumps(item))
    assert not compare(c,'micro4')['passed']
