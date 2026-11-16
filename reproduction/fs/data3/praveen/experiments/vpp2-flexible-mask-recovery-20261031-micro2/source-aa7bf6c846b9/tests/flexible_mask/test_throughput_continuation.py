import copy
import json
from pathlib import Path
import pytest
from vpp2.flexible_mask.config import validate
from vpp2.flexible_mask.continuation import scientific_recipe,validate_parent
from vpp2.full_mask.panel import file_hash


def configuration():
    return json.loads((Path(__file__).resolve().parents[2]/'configs/flexible_mask/full30k.json').read_text())


def test_optimization_preserves_scientific_recipe():
    old=configuration();new=copy.deepcopy(old)
    new['train']['micro_batch']=2
    new['execution']={'fill_uninitialized_memory':False}
    new['root']='/new';new['resume_from']={'root':'/old'}
    validate(new)
    assert scientific_recipe(new)==scientific_recipe(old)
    assert new['train']['global_batch']==256
    for key in ('teacher_video_weight','kv_weight','action_gt_weight','frozen_video'):
        assert new['train'][key]==old['train'][key]


@pytest.mark.parametrize('value',[0,3,4,True,2.0])
def test_unverified_microbatch_rejected(value):
    c=configuration();c['train']['micro_batch']=value
    with pytest.raises(ValueError,match='microbatch'):validate(c)


@pytest.mark.parametrize('key,value',[('video_lr',2e-6),('teacher_video_weight',0.),('kv_weight',.1),('global_batch',288)])
def test_performance_migration_cannot_hide_scientific_changes(key,value):
    a=configuration();b=copy.deepcopy(a);b['train'][key]=value
    assert scientific_recipe(a)!=scientific_recipe(b)


def parent_setup(tmp_path):
    old=configuration();parent=tmp_path/'old';parent.mkdir()
    old['root']=str(parent);(parent/'config.json').write_text(json.dumps(old))
    tag='update_000033';folder=parent/'mask/checkpoints'/tag;folder.mkdir(parents=True)
    receipt=dict(identity='previous',stage='mask',update=33,checkpoint_identity='pinned')
    (folder/'COMPLETE.json').write_text(json.dumps(receipt))
    (parent/'inputs.json').write_text(json.dumps(dict(identity='previous',data='same')))
    new=copy.deepcopy(old);root=tmp_path/'new';root.mkdir();new['root']=str(root)
    new['train']['micro_batch']=2;new['execution']={'fill_uninitialized_memory':False}
    (root/'inputs.json').write_text(json.dumps(dict(identity='new',data='same')))
    new['resume_from']=dict(root=str(parent),identity='previous',tag=tag,
        config_sha256=file_hash(parent/'config.json'),receipt_sha256=file_hash(folder/'COMPLETE.json'))
    return new,folder


def test_parent_checkpoint_is_pinned_and_inputs_identical(tmp_path):
    c,folder=parent_setup(tmp_path)
    _,r=validate_parent(c,'mask');assert r['update']==33
    (folder/'COMPLETE.json').write_text('{}')
    with pytest.raises(ValueError,match='receipt changed'):validate_parent(c,'mask')


def test_parent_data_or_objective_changes_fail_closed(tmp_path):
    c,_=parent_setup(tmp_path);c['train']['teacher_video_weight']=0
    with pytest.raises(ValueError,match='scientific recipe'):validate_parent(c,'mask')
    c['train']['teacher_video_weight']=.05
    (Path(c['root'])/'inputs.json').write_text(json.dumps(dict(identity='new',data='different')))
    with pytest.raises(ValueError,match='inputs changed'):validate_parent(c,'mask')


@pytest.mark.parametrize('initially_cached',[set(),{1},{1,2}])
def test_mixed_cache_batches_keep_single_window_encoder_path(monkeypatch,initially_cached):
    from types import SimpleNamespace
    from vpp2.flexible_mask.data import prepare_cached
    from vpp2.adaptive_mask import data
    saved={i:{'conditioning':i*7} for i in initially_cached};calls=[]
    cache=SimpleNamespace(load=lambda key:saved.get(key));reader=SimpleNamespace(runtime_cache=cache)
    def native(model,reader,rows,positions,seed,strata):
        ids=[r['id'] for r in rows];calls.append((ids,positions,seed,strata))
        if len(ids)>1:assert all(i in saved for i in ids)
        for i in ids:saved[i]={'conditioning':i*7}
        return tuple(saved[i]['conditioning'] for i in ids),tuple(positions),tuple(strata)
    monkeypatch.setattr(data,'prepare',native)
    result=prepare_cached(None,reader,[{'id':1},{'id':2}],[901,902],100008,[2,0])
    assert result==((7,14),(901,902),(2,0))
    assert calls[-1]==([1,2],[901,902],100008,[2,0])
    assert all(len(x[0])==1 for x in calls[:-1])
    assert reader.batch_cache_statistics['singly_encoded_windows']==2-len(initially_cached)
    assert reader.batch_cache_statistics['mixed_batches']==int(len(initially_cached)==1)
    before=len(calls)
    assert prepare_cached(None,reader,[{'id':1},{'id':2}],[903,904],100008,[1,3])==((7,14),(903,904),(1,3))
    assert len(calls)==before+1


def test_cache_failure_does_not_allow_partial_batch(monkeypatch):
    from types import SimpleNamespace
    from vpp2.flexible_mask.data import prepare_cached
    from vpp2.adaptive_mask import data
    reader=SimpleNamespace(runtime_cache=SimpleNamespace(load=lambda key:None))
    monkeypatch.setattr(data,'prepare',lambda *args:None)
    with pytest.raises(RuntimeError,match='not populated'):
        prepare_cached(None,reader,[{'id':1},{'id':2}],[0,1],100008,[0,1])
