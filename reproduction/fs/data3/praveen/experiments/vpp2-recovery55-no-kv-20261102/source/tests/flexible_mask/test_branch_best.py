"""Selection must retain real weights at every validation, including non-save steps."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from vpp2.flexible_mask import recovery_best as best, recovery_export as export
from vpp2.flexible_mask.selection import aggregate
from vpp2.flexible_mask.preflight import fingerprint
from vpp2.flexible_mask.rng import rng_state
from vpp2.full_mask.panel import file_hash,digest


def evidence(tmp_path,update,error=1.,teacher=.1):
    rows=[dict(window_id=t*4+i,episode=t*4+i,frame=0,executed_steps=24,
               task=f'task{t:02d}',suite=f'suite{t%5}',student_error=error,teacher_error=teacher)
          for t in range(35) for i in range(4)]
    result=dict(update=update,full=False,native_10_step_sampler=True,simulation=False,
                rows=rows,actions=aggregate(rows))
    path=tmp_path/f'validation_{update}.json';path.write_text(json.dumps(result))
    return result,best.candidate(result,100,path)


def fake_artifact(tmp_path,update):
    p=tmp_path/f'policy_{update}';p.mkdir(exist_ok=True)
    return dict(kind='native_policy',path=str(p),receipt_sha256='receipt',initial_export_sha256='export')


def setup_runtime(tmp_path,monkeypatch):
    monkeypatch.setattr(best.dist,'barrier',lambda:None)
    monkeypatch.setattr('vpp2.flexible_mask.config.identity',lambda c:'identity')
    return SimpleNamespace(rank=0,physical_parameters=100,config={'root':str(tmp_path)})


def test_gt_first_ranking_does_not_prioritize_teacher_error_ratio(tmp_path):
    _,a=evidence(tmp_path,250,1.,1e-8)
    _,b=evidence(tmp_path,500,2.,2.)
    assert best.select([b,a])[0]['update']==250
    assert best.key(a)[0]==1.


def test_same_gt_metric_prefers_earlier_equal_size_candidate(tmp_path):
    _,a=evidence(tmp_path,250);_,b=evidence(tmp_path,500)
    assert [x['update'] for x in best.select([b,a])]==[250,500]


def test_rejects_changed_or_incomplete_validation_panel(tmp_path):
    result,a=evidence(tmp_path,250)
    result['rows'].pop()
    with pytest.raises(ValueError,match='35 tasks'):best.candidate(result,100,a['validation_path'])
    _,b=evidence(tmp_path,500);b['window_identity']='different'
    with pytest.raises(ValueError,match='different windows'):best.select([a,b])


def test_update_250_improvement_exports_without_restart_checkpoint(tmp_path,monkeypatch):
    rt=setup_runtime(tmp_path,monkeypatch)
    _,old=evidence(tmp_path,55,2.);old['artifact']=fake_artifact(tmp_path,55)
    result,new=evidence(tmp_path,250,1.)
    calls=[]
    def publish(runtime,initial,value):
        calls.append(value['update']);return fake_artifact(tmp_path,value['update'])
    monkeypatch.setattr(export,'publish_best',publish)
    selected=best.observe(rt,'initial',result,[old],new['validation_path'])
    assert calls==[250]
    assert [v['update'] for v in selected]==[250,55]
    assert not (tmp_path/'recovery/checkpoints/update_000250').exists()
    saved=json.loads((tmp_path/'recovery/best_models.json').read_text())
    assert saved['best'][0]['artifact']['path']==str(tmp_path/'policy_250')


def test_non_improvement_does_not_export_or_replace_top_two(tmp_path,monkeypatch):
    rt=setup_runtime(tmp_path,monkeypatch)
    previous=[]
    for u,e in [(55,1.),(250,2.)]:
        _,v=evidence(tmp_path,u,e);v['artifact']=fake_artifact(tmp_path,u);previous.append(v)
    result,current=evidence(tmp_path,750,3.)
    monkeypatch.setattr(export,'publish_best',lambda *a:pytest.fail('Non-best must not export'))
    selected=best.observe(rt,'initial',result,previous,current['validation_path'])
    assert selected==previous
    assert json.loads((tmp_path/'recovery/best_models.json').read_text())['latest_validation']['update']==750


def test_failed_export_leaves_existing_best_manifest_intact(tmp_path,monkeypatch):
    rt=setup_runtime(tmp_path,monkeypatch)
    p=tmp_path/'recovery/best_models.json';p.parent.mkdir();p.write_text('{"previous":true}')
    result,value=evidence(tmp_path,250)
    def fail(*args):raise RuntimeError('disk failure')
    monkeypatch.setattr(export,'publish_best',fail)
    with pytest.raises(RuntimeError,match='disk failure'):best.observe(rt,'initial',result,[],value['validation_path'])
    assert p.read_text()=='{"previous":true}'


def test_pruning_preserves_restart_references_and_uncommitted_exports(tmp_path,monkeypatch):
    rt=setup_runtime(tmp_path,monkeypatch);root=tmp_path/'recovery/best_models';root.mkdir(parents=True)
    for u in (250,500,750,1000):
        p=root/f'update_{u:06d}';p.mkdir()
        (p/'COMPLETE.json').write_text(json.dumps(dict(identity='identity',weights_only=True,update=u)))
    partial=root/'.update_001250.partial';partial.mkdir()
    checkpoint=tmp_path/'recovery/checkpoints/update_000500';checkpoint.mkdir(parents=True)
    (checkpoint/'COMPLETE.json').write_text('{"checkpoint_identity":"saved"}')
    current=[dict(artifact=dict(path=str(root/'update_000750')))]
    best.prune_exports(rt.config,current,750)
    assert (root/'update_000250').exists()  # missing sidecar cannot authorize deletion
    (checkpoint/'BEST_MODELS.json').write_text(json.dumps(dict(checkpoint_identity='saved',paths=[str(root/'update_000500')])))
    best.prune_exports(rt.config,current,750)
    assert not (root/'update_000250').exists()
    assert all((root/f'update_{u:06d}').exists() for u in (500,750,1000))
    assert partial.exists()


def test_best_export_preserves_rng_optimizer_and_live_weights(tmp_path,monkeypatch):
    setup_runtime(tmp_path,monkeypatch)
    teacher=tmp_path/'teacher';teacher.mkdir();torch.save({'step':100000,'policy':torch.ones(1)},teacher/'action.pt')
    initial=tmp_path/'initial';(initial/'native').mkdir(parents=True)
    for name in ['dataset_stats.json','compact_architecture.json']:(initial/'native'/name).write_text('{}')
    weights={'video.weight':torch.tensor([[1.,2.],[3.,4.]])}
    model=torch.nn.Linear(2,2);optimizer=torch.optim.AdamW(model.parameters(),lr=.1)
    model(torch.ones(1,2)).sum().backward();optimizer.step();optimizer.zero_grad()
    class Engine:
        def _zero3_consolidated_16bit_state_dict(self,exclude_frozen_parameters):
            torch.rand(3)  # even an incidental RNG draw must be isolated
            return weights
    rt=SimpleNamespace(config={'root':str(tmp_path),'teacher_bundle':str(teacher)},rank=0,engine=Engine(),recovery_export_sha256='initialhash')
    monkeypatch.setattr(export,'read_export_manifest',lambda *a,**k:dict(tensors={'weight':{'shape':[2,2]}},parameters=4,mask_identity='mask'))
    monkeypatch.setattr(export,'identity',lambda c:'identity')
    result,value=evidence(tmp_path,250)
    value['parameters']=4
    before=fingerprint(dict(rng=rng_state(),optimizer=optimizer.state_dict(),model=model.state_dict(),video=weights))
    artifact=export.publish_best(rt,initial,value)
    after=fingerprint(dict(rng=rng_state(),optimizer=optimizer.state_dict(),model=model.state_dict(),video=weights))
    assert before==after
    path=Path(artifact['path']);value['artifact']=artifact
    best.verify_artifact(value)
    assert torch.equal(torch.load(path/'video.pt',weights_only=True)['dit']['weight'],weights['video.weight'])
    assert torch.load(path/'action.pt',weights_only=True)['step']==250
    assert not (tmp_path/'recovered_policy.json').exists()  # best export is not end-of-training
    assert not list(path.glob('*optim*'))
    assert export.publish_best(rt,initial,value)==artifact  # idempotent exact replay
    weights['video.weight'].add_(1)
    with pytest.raises(ValueError,match='differs'):export.publish_best(rt,initial,value)


