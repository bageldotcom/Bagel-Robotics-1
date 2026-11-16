import json
import pytest
import torch
from vpp2.flexible_mask.metrics import DeferredMetrics
from vpp2.flexible_mask.report import Reporter


def legacy(records):
    rec=[];tasks=[]
    for metrics,details,rows in records:
        keys=list(metrics)
        values=torch.stack([torch.as_tensor(metrics[k]).detach().float() for k in keys]).cpu().tolist()
        rec.append(dict(zip(keys,values)))
        action=torch.stack([details[k] for k in ('student_error','teacher_error','reliability')],-1).cpu().tolist()
        tasks.extend(dict(task=r['task'],suite=r['suite'],student_error=action[i][0],
                          teacher_error=action[i][1],reliability=action[i][2]) for i,r in enumerate(rows))
    return {k:sum(x[k] for x in rec)/len(rec) for k in rec[0]},tasks


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('micro',[1,2])
def test_logging_preserves_records_and_gradients_without_microbatch_transfers(monkeypatch,dtype,micro):
    torch.manual_seed(711)
    records=[];parameter=torch.tensor(.25,requires_grad=True)
    for i in range(16):
        values=torch.randn(micro,3,dtype=dtype)
        metrics={'gt_action':parameter*(i+1),'late_kv_raw':torch.tensor(1e20 if i%3==0 else -1e20 if i%3==2 else 1.),
                 'kv/layer_39_k':torch.tensor(i/7,dtype=dtype),'weight':.05}
        details={k:values[:,j] for j,k in enumerate(('student_error','teacher_error','reliability'))}
        rows=[dict(task='task'+str(i%3),suite='suite'+str(j%2)) for j in range(micro)]
        records.append((metrics,details,rows))
    expected=legacy(records)
    original_cpu=torch.Tensor.cpu;calls=[]
    def count_cpu(tensor,*args,**kwargs):
        calls.append(tensor.shape)
        return original_cpu(tensor,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'cpu',count_cpu)
    collector=DeferredMetrics(torch.device('cpu'))
    before=torch.get_rng_state().clone()
    for metrics,details,rows in records:
        collector.append(metrics,details,rows)
        metrics['gt_action'].backward()
    assert calls==[]
    assert all(not t.requires_grad and t.grad_fn is None for t in collector.scalars+collector.actions)
    assert torch.equal(before,torch.get_rng_state())
    assert float(parameter.grad)==sum(range(1,17))
    assert collector.finish()==expected
    assert len(calls)==2
    assert not collector.scalars and not collector.actions and not collector.labels


def test_metric_and_window_contracts_fail_closed():
    c=DeferredMetrics(torch.device('cpu'))
    d={k:torch.zeros(1) for k in ('student_error','teacher_error','reliability')}
    rows=[dict(task='a',suite='memory')]
    with pytest.raises(ValueError,match='empty'):c.finish()
    c.append({'a':torch.tensor(1.)},d,rows)
    with pytest.raises(ValueError,match='columns'):c.append({'b':torch.tensor(1.)},d,rows)
    with pytest.raises(ValueError,match='labels'):c.append({'a':torch.tensor(1.)},d,rows*2)


class FakeRun:
    def __init__(self):self.records=[];self.finished=False
    def log(self,row,step):self.records.append((step,dict(row)))
    def finish(self):self.finished=True


def test_moderate_remote_logs_keep_every_local_record_and_flush_final(tmp_path,capsys):
    reporter=Reporter({},tmp_path,'mask',enabled=False);reporter.run=FakeRun()
    expected=[]
    for u in range(1,18):
        row=dict(update=u,stage='mask',gt_action=u/100,**{'kv/layer_00_k':u/200})
        expected.append(row);reporter.write(row)
    assert [step for step,_ in reporter.run.records]==[1,5,10,15]
    reporter.close()
    assert [step for step,_ in reporter.run.records]==[1,5,10,15,17]
    assert [json.loads(x) for x in (tmp_path/'metrics.jsonl').read_text().splitlines()]==expected
    assert [json.loads(x)['update'] for x in capsys.readouterr().out.splitlines()]==[1,10,17]
    assert reporter.run.finished


def test_calibration_logs_are_not_thinned(tmp_path,capsys):
    reporter=Reporter({},tmp_path,'mask',enabled=False);reporter.run=FakeRun()
    for u in range(193,201):
        reporter.write(dict(update=u,**{'calibration/hidden/gt_norm':.3}))
    reporter.close()
    assert [step for step,_ in reporter.run.records]==list(range(193,201))
    assert len(capsys.readouterr().out.splitlines())==8


def test_reporter_close_without_records(tmp_path):
    reporter=Reporter({},tmp_path,'mask',enabled=False);reporter.run=FakeRun()
    reporter.close()
    assert reporter.run.finished and reporter.run.records==[]
