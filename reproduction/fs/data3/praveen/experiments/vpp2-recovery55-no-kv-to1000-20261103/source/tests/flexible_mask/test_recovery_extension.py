"""Continuation must preserve the recipe, exact resume state, and best two policies."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

from vpp2.flexible_mask import checkpoint, config, low_lr, recovery_best, recovery_export, recovery_subset
from vpp2.flexible_mask import recovery_extension as extension
from vpp2.flexible_mask.preflight import fingerprint
from vpp2.flexible_mask.rng import rng_state
from vpp2.flexible_mask.selection import aggregate


def recipe():
    return json.loads((Path(__file__).resolve().parents[3]/'config.json').read_text())


def evidence(update, error):
    rows=[dict(window_id=4*t+i,episode=4*t+i,frame=0,executed_steps=24,task=f't{t}',suite=f's{t%5}',
               student_error=error,teacher_error=.1,student_teacher_error=error/2,
               student_components=[error]*3,teacher_components=[.1]*3)
          for t in range(35) for i in range(4)]
    return dict(update=update,full=False,simulation=False,native_10_step_sampler=True,
                rows=rows,actions=aggregate(rows),action_agreement={'overall':error/2})


def test_extension_only_changes_duration_locations_and_reporting():
    c=recipe();config.validate(c)
    p=Path(c['recovery_extension']['parent_root'])
    old=json.loads((p/'config.json').read_text())
    assert extension.canonical_recipe(c)==extension.canonical_recipe(old)
    assert c['recovery_extension']['target_update']==1000
    assert c['recovery_lr_branch']['schedule_endpoint']==2399
    assert all(low_lr.learning_rate(c,u)==low_lr.learning_rate(old,u) for u in range(251,1001))
    assert low_lr.learning_rate(c,251)<1e-6
    assert low_lr.learning_rate(c,251)>.99999e-6
    assert low_lr.learning_rate(c,1000)>1e-7
    assert recovery_subset.sample_position(c,250*256)==(250-55)*256
    assert recovery_subset.sample_position(c,999*256)==(999-55)*256


@pytest.mark.parametrize('field,value',[('target_update',1001),('parent_update',55),('parent_tag','update_000200'),
                                       ('condition','always'),('gradient_updates',[])])
def test_invalid_endpoint_or_parent_is_rejected(field,value):
    c=recipe();c['recovery_extension'][field]=value
    with pytest.raises(ValueError):config.validate(c)


@pytest.mark.parametrize('field,value',[('kv_weight',.05),('teacher_video_weight',0.),('global_batch',128),
                                       ('frozen_action',False),('action_gt_weight',.5)])
def test_extension_cannot_change_objectives_or_batch(field,value):
    c=recipe();c['train'][field]=value
    with pytest.raises(ValueError):config.validate(c)


def test_parent_condition_requires_actual_matched_gt_improvement():
    old=evidence(250,1.)
    assert extension.assess_validation(evidence(250,.9),old)['relative_improvement']==pytest.approx(.1)
    for error in (1.,1.1):
        with pytest.raises(RuntimeError,match='withheld'):extension.assess_validation(evidence(250,error),old)
    bad=evidence(250,.9);bad['rows'][0]['frame']=1
    with pytest.raises(ValueError,match='different'):extension.assess_validation(bad,old)
    bad=evidence(250,.9);bad['rows'].pop()
    with pytest.raises(ValueError):extension.assess_validation(bad,old)


def test_training_arithmetic_and_export_retention_are_unchanged():
    c=recipe();parent=Path(c['recovery_extension']['parent_root'])/'source/src/vpp2/flexible_mask'
    child=Path(__file__).resolve().parents[2]/'src/vpp2/flexible_mask'
    for name in ('runtime.py','losses.py','checkpoint.py','data.py','rng.py',
                 'recovery_gradients.py','recovery_export.py','recovery_best.py'):
        assert (parent/name).read_bytes()==(child/name).read_bytes(),name


def test_resume250_preserves_optimizer_rng_cursor_and_best_records(tmp_path,monkeypatch):
    monkeypatch.setattr(checkpoint.dist,'barrier',lambda:None)
    monkeypatch.setattr(checkpoint.dist,'all_gather_object',lambda out,v:out.__setitem__(0,v))
    monkeypatch.setattr(config,'identity',lambda c:'child')
    parent=tmp_path/'parent';child=tmp_path/'child'
    (parent/'recovery').mkdir(parents=True);child.mkdir()
    c=recipe();c['root']=str(child);c['recovery_extension']['parent_root']=str(parent)
    c['recovery_extension']['parent_identity']='parent'
    old=copy.deepcopy(c);old.pop('recovery_extension');old['root']=str(parent)
    (parent/'config.json').write_text(json.dumps(old))
    class Engine:
        def __init__(self):
            self.module=torch.nn.Linear(2,2);self.optimizer=torch.optim.AdamW(self.module.parameters(),lr=1e-6)
            self.module(torch.ones(1,2)).square().mean().backward();self.optimizer.step();self.optimizer.zero_grad()
            self.global_steps=250
        def save_checkpoint(self,root,tag,client_state,save_latest):
            p=Path(root)/tag;p.mkdir(parents=True)
            torch.save(dict(model=self.module.state_dict(),optimizer=self.optimizer.state_dict(),
                            state=client_state,steps=self.global_steps),p/'rank.pt')
        def load_checkpoint(self,root,tag,**kwargs):
            d=torch.load(Path(root)/tag/'rank.pt',weights_only=False)
            self.module.load_state_dict(d['model']);self.optimizer.load_state_dict(d['optimizer']);self.global_steps=d['steps']
            return str(Path(root)/tag),d['state']
    rt=SimpleNamespace(rank=0,world=1,engine=Engine(),masks=None,mask_optimizer=None,recovery_export_sha256='export')
    state=dict(identity='parent',update=250,stage='recovery',best=[],history=[],coefficient={},
               **low_lr.state_fields(old),**recovery_subset.state_fields(old,250))
    checkpoint.save(rt,parent/'recovery',state)
    receipt=json.loads((parent/'recovery/checkpoints/update_000250/COMPLETE.json').read_text())
    monkeypatch.setattr(extension,'bind_parent',lambda c:dict(checkpoint_identity=receipt['checkpoint_identity'],best_paths=[]))
    monkeypatch.setattr('vpp2.flexible_mask.recovery_gradients.adam_state',lambda engine:engine.optimizer.state_dict())
    before=fingerprint(dict(weights=rt.engine.module.state_dict(),adam=rt.engine.optimizer.state_dict(),rng=rng_state()))
    with torch.no_grad():
        for p in rt.engine.module.parameters():p.add_(10.)
    torch.manual_seed(23);rt.engine.global_steps=0
    restored=extension.load_parent(rt,c)
    after=fingerprint(dict(weights=rt.engine.module.state_dict(),adam=rt.engine.optimizer.state_dict(),rng=rng_state()))
    assert before==after
    assert restored['update']==250 and restored['cursor']==64000
    assert restored['recovery_subset_cursor']==49920 and restored['identity']=='child'
    assert restored['extension_parent_checkpoint_identity']==receipt['checkpoint_identity']
    extension.check_state(c,restored)
    bad=copy.deepcopy(restored);bad['recovery_subset_cursor']=0
    with pytest.raises(ValueError):extension.check_state(c,bad)
    bad=copy.deepcopy(restored);bad['update']=1001
    with pytest.raises(ValueError):extension.check_state(c,bad)
    assert (parent/'recovery/checkpoints/update_000250/COMPLETE.json').exists()


def test_restored_validation_checks_each_window_and_teacher_agreement(tmp_path):
    c=recipe();c['root']=str(tmp_path/'child');c['recovery_extension']['parent_root']=str(tmp_path/'parent')
    p=Path(c['recovery_extension']['parent_root'])/'recovery/diagnostics/native_000250.json'
    p.parent.mkdir(parents=True);d=evidence(250,1.);p.write_text(json.dumps(d))
    runtime=SimpleNamespace(config=c,rank=0)
    extension.verify_restored_validation(runtime,d)
    bad=copy.deepcopy(d);bad['rows'][0]['student_teacher_error']*=1.1
    with pytest.raises(ValueError,match='predictions changed'):extension.verify_restored_validation(runtime,bad)


def test_best_two_span_parent_and_new_checkpoints(tmp_path,monkeypatch):
    monkeypatch.setattr(recovery_best.dist,'barrier',lambda:None)
    monkeypatch.setattr(config,'identity',lambda c:'child')
    parent=tmp_path/'parent';child=tmp_path/'child';parent.mkdir();child.mkdir()
    rt=SimpleNamespace(rank=0,physical_parameters=100,config={'root':str(child)})
    def candidate(update,error,root):
        result=evidence(update,error);path=root/f'validation_{update}.json';path.write_text(json.dumps(result))
        item=recovery_best.candidate(result,100,path)
        return result,item,path
    best=[]
    for u,e in [(200,.7),(250,.6)]:
        _,item,_=candidate(u,e,parent);p=parent/f'policy_{u}';p.mkdir()
        item['artifact']=dict(kind='native_policy',path=str(p));best.append(item)
    def publish(runtime,initial,item):
        p=child/f'policy_{item["update"]}';p.mkdir()
        return dict(kind='native_policy',path=str(p))
    monkeypatch.setattr(recovery_export,'publish_best',publish)
    for u,e in [(300,.8),(350,.5),(400,.55),(450,.9)]:
        result,item,path=candidate(u,e,child)
        best=recovery_best.observe(rt,'initial',result,best,path)
        expected=[250,200] if u==300 else [350,250] if u==350 else [350,400]
        assert [x['update'] for x in best]==expected
        manifest=json.loads((child/'recovery/best_models.json').read_text())
        assert [x['update'] for x in manifest['best']]==expected
        assert len(best)==2
    assert (parent/'policy_200').exists() and (parent/'policy_250').exists()
