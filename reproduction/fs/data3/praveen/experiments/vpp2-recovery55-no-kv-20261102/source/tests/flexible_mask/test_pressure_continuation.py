"""Pressure changes must be explicit, bounded, restartable and isolated."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vpp2.flexible_mask.coefficient import Coefficient
from vpp2.flexible_mask.config import validate
from vpp2.flexible_mask.pressure import (
    validate as validate_pressure, weight, validate_recipe_change, validate_parent_state,
)
from vpp2.flexible_mask.preflight import fingerprint

BASE = 0.011596363996894921


def configuration():
    c = json.loads((Path(__file__).resolve().parents[2] / 'configs/flexible_mask/full30k.json').read_text())
    c['pressure_continuation'] = dict(
        schema='vpp2-size-pressure-ramp-v1', start_update=1250, ramp_updates=250,
        target_multiplier=2., base_coefficient=BASE, absolute_cap=2*BASE,
        target_video_parameters=10_000_000_000, automatic_second_increase=False)
    return c


def coefficient():
    return Coefficient(measured=BASE, applied=BASE, resolved_update=200)


def test_pressure_disabled_is_identical_to_original_schedule():
    c = configuration(); c.pop('pressure_continuation')
    k = coefficient()
    assert [weight(k,c,u) for u in range(1,3001)] == [k.weight(u) for u in range(1,3001)]


def test_ramp_has_no_boundary_jump_and_never_exceeds_twofold_cap():
    c = configuration(); validate(c); k = coefficient()
    assert weight(k,c,1250) == BASE
    assert weight(k,c,1251) == pytest.approx(BASE*1.004)
    assert weight(k,c,1375) == pytest.approx(BASE*1.5)
    assert weight(k,c,1500) == 2*BASE
    assert weight(k,c,3000) == 2*BASE
    values = [weight(k,c,u) for u in range(1250,3001)]
    assert all(a <= b for a,b in zip(values,values[1:]))


@pytest.mark.parametrize('key,value', [
    ('start_update', True), ('start_update', 999), ('start_update', 2000),
    ('ramp_updates', 0), ('target_multiplier', 4.),
    ('base_coefficient', 0.), ('base_coefficient', float('nan')),
    ('absolute_cap', 1.), ('automatic_second_increase', True),
    ('target_video_parameters', 8_000_000_000),
])
def test_invalid_pressure_changes_are_rejected(key,value):
    c = configuration(); c['pressure_continuation'][key] = value
    with pytest.raises(ValueError): validate(c)


def test_changed_or_unresolved_parent_coefficient_is_rejected():
    c=configuration(); k=coefficient()
    k.applied=BASE*1.01
    with pytest.raises(ValueError,match='saved calibrated'):weight(k,c,1251)
    with pytest.raises(ValueError,match='saved calibrated'):weight(Coefficient(),c,1251)


@pytest.mark.parametrize('field,value', [
    ('teacher_video_weight', 0.), ('kv_weight', .1), ('action_gt_weight', .5),
    ('micro_batch', 2), ('global_batch', 128), ('video_lr', 2e-6),
    ('frozen_video', True), ('max_updates', 5000),
])
def test_pressure_continuation_cannot_hide_other_changes(field,value):
    new=configuration(); old=copy.deepcopy(new); old.pop('pressure_continuation')
    new['train'][field]=value
    with pytest.raises(ValueError,match='another scientific or execution'):
        validate_recipe_change(new,old)


def test_only_explicit_pressure_lineage_and_display_name_changes_allowed():
    new=configuration(); old=copy.deepcopy(new); old.pop('pressure_continuation')
    new['root']='/different/root'; new['resume_from']={'tag':'update_001250'}
    new['wandb']['name']='pressure-2x'
    validate_recipe_change(new,old)
    with pytest.raises(ValueError,match='Stacking'):validate_recipe_change(new,new)


def test_parent_update_and_cursor_cannot_replay_or_skip_training():
    c=configuration()
    state=dict(update=1250,cursor=1250*256,coefficient=coefficient().state_dict())
    validate_parent_state(state,c)
    for patch in [dict(update=1251),dict(cursor=1250*256+1)]:
        with pytest.raises(ValueError,match='pinned checkpoint'):validate_parent_state(state|patch,c)


def test_actual_gradient_delta_is_only_the_size_term():
    c=configuration(); k=coefficient()
    video=torch.nn.Linear(3,3)
    score=torch.nn.Parameter(torch.tensor([-.12,.03,.19]))
    x=torch.tensor([[1.,-2.,3.]])
    def loss(w):
        gate=torch.sigmoid(10*score)
        prediction=video(x)*gate
        task=(prediction-torch.tensor([[.1,.2,.3]])).square().mean()
        return task+w*gate.mean()
    params=(score,*video.parameters())
    a=torch.autograd.grad(loss(weight(k,c,1250)),params)
    b=torch.autograd.grad(loss(weight(k,c,1500)),params)
    expected=torch.autograd.grad(BASE*torch.sigmoid(10*score).mean(),score)[0]
    torch.testing.assert_close(b[0]-a[0],expected,atol=3e-7,rtol=3e-5)
    for left,right in zip(a[1:],b[1:]):assert torch.equal(left,right)


def test_interrupted_adam_resume_preserves_ramp_optimizer_rng_and_parameters(tmp_path):
    c=configuration(); k=coefficient()
    torch.manual_seed(100008)
    video=torch.nn.Linear(3,3)
    score=torch.nn.Parameter(torch.tensor([-.12,.03,.19]))
    optimizer=torch.optim.AdamW([score,*video.parameters()],lr=1e-3)
    def updates(lo,hi):
        for u in range(lo,hi+1):
            x=torch.randn(4,3)
            y=video(x)*torch.sigmoid(10*score)
            loss=(y-x*.2).square().mean()+weight(k,c,u)*torch.sigmoid(10*score).mean()
            loss.backward();optimizer.step();optimizer.zero_grad()
    updates(1248,1375)
    saved=dict(video=copy.deepcopy(video.state_dict()),masks=score.detach().clone(),
               optimizer=copy.deepcopy(optimizer.state_dict()),rng=torch.get_rng_state(),
               coefficient=k.state_dict(),update=1375)
    torch.save(saved,tmp_path/'state.pt')
    updates(1376,1503)
    expected=fingerprint((video.state_dict(),score,optimizer.state_dict(),torch.get_rng_state()))
    state=torch.load(tmp_path/'state.pt',weights_only=False)
    video.load_state_dict(state['video'])
    with torch.no_grad():score.copy_(state['masks'])
    optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng'])
    k=Coefficient.from_state(state['coefficient'],1.)
    updates(state['update']+1,1503)
    assert fingerprint((video.state_dict(),score,optimizer.state_dict(),torch.get_rng_state()))==expected


def test_gpu_probe_logic_is_nonmutating_on_cpu_fixture(tmp_path,monkeypatch):
    from vpp2.flexible_mask import pressure
    from vpp2.flexible_mask import config as config_module
    c=configuration(); c['root']=str(tmp_path)
    from vpp2.adaptive_mask.geometry import Geometry, ParameterInventory
    from vpp2.flexible_mask.controller import Masks, SoftCost
    from vpp2.models.wan21_14b.wan_video_dit import WanVideoDiT
    geometry=Geometry(hidden=80,ffn=128,layers=3,heads=4,head_dim=8,pack=2,
                      hidden_floor=76,ffn_floor=16,head_floor=1)
    with torch.device('meta'):
        model=WanVideoDiT(hidden_dim=80,ffn_dim=128,num_layers=3,num_heads=4,attn_head_dim=8,
                          freq_dim=8,text_dim=10,in_dim=8,out_dim=2,cross_attn_norm=True)
    # Use the production cost and mask interfaces, not an invented test stub.
    cost=SoftCost(ParameterInventory.from_model(model,geometry))
    masks=Masks(geometry,
        dict(hidden_scores=torch.arange(80.),ffn_scores=torch.arange(256.).reshape(2,128),
             cross_scores=torch.arange(8.).reshape(2,4)),
        dict(hidden_scores=76,ffn_scores=64,cross_scores=3))
    with torch.no_grad():
        for p in masks.parameters():p.mul_(.1)
    optimizer=torch.optim.AdamW(masks.parameters(),lr=.01)
    sum(p.square().mean() for p in masks.parameters()).backward()
    optimizer.step();optimizer.zero_grad()
    runtime=SimpleNamespace(masks=masks,mask_optimizer=optimizer,world=1,rank=0,cost=cost)
    monkeypatch.setattr(pressure,'require_runtime_evidence',lambda c:None)
    monkeypatch.setattr(config_module,'identity',lambda c:'cpu-unit-test-only')
    monkeypatch.setattr(torch.distributed,'all_gather_object',lambda out,value:out.__setitem__(0,value))
    monkeypatch.setattr(torch.distributed,'barrier',lambda:None)
    pressure.verify_gpu(runtime,c,dict(update=1250),coefficient())
    result=json.loads((tmp_path/'checks/pressure_gpu.json').read_text())
    assert result['passed'] and result['identity']=='cpu-unit-test-only'
    assert all(p.grad is None for p in masks.parameters())


def test_changed_pressure_cannot_reuse_old_stability_interval():
    from vpp2.flexible_mask.selection import readiness
    from vpp2.adaptive_mask.geometry import Geometry
    g=Geometry(hidden=8,ffn=16,layers=2,heads=2,head_dim=8,pack=2,
               hidden_floor=6,ffn_floor=4,head_floor=1)
    c=configuration(); k=coefficient()
    def row(u):
        return dict(update=u, selection=g.full(), parameters=100,
                    regularizer_state=[10.,.01,.01,weight(k,c,u)], gradient_active=True,
                    actions={'task/test':dict(student_error=.1,teacher_error=.05),
                             'suite/test':dict(student_error=.1,teacher_error=.05)})
    changing=[row(u) for u in (1250,1500,1750,2000,2250)]
    result=readiness(changing,g,2250,2)
    assert 'temperature_or_regularizer_still_changing' in result['reasons']
    constant=[row(u) for u in (1500,1750,2000,2250,2500)]
    assert readiness(constant,g,2500,2)['status']=='ready_for_recovery'


def test_parent_lineage_checks_pressure_and_keeps_existing_checkpoint_contract(tmp_path):
    from vpp2.flexible_mask.continuation import validate_parent
    from vpp2.full_mask.panel import file_hash
    c=configuration(); old=copy.deepcopy(c); old.pop('pressure_continuation')
    parent=tmp_path/'old'; child=tmp_path/'new'
    parent.mkdir();child.mkdir()
    old['root']=str(parent);c['root']=str(child)
    (parent/'config.json').write_text(json.dumps(old))
    tag='update_001250'; folder=parent/'mask/checkpoints'/tag;folder.mkdir(parents=True)
    (folder/'COMPLETE.json').write_text(json.dumps(dict(identity='parent-id',stage='mask',
        update=1250,checkpoint_identity='checkpoint-id')))
    (parent/'inputs.json').write_text(json.dumps(dict(identity='parent-id',data='unchanged')))
    (child/'inputs.json').write_text(json.dumps(dict(identity='child-id',data='unchanged')))
    c['resume_from']=dict(root=str(parent),tag=tag,identity='parent-id',
        config_sha256=file_hash(parent/'config.json'),receipt_sha256=file_hash(folder/'COMPLETE.json'))
    assert validate_parent(c,'mask')[1]['update']==1250
    c['train']['micro_batch']=2
    with pytest.raises(ValueError,match='another scientific or execution'):validate_parent(c,'mask')


def test_resume_between_regular_checkpoints_does_not_rewind_updates():
    c=configuration();c['pressure_continuation']['start_update']=1251
    validate(c)
    state=dict(update=1251,cursor=1251*256,coefficient=coefficient().state_dict())
    validate_parent_state(state,c)
    assert weight(coefficient(),c,1251)==BASE
    assert weight(coefficient(),c,1501)==2*BASE


def publication_fixture(tmp_path,monkeypatch):
    from vpp2.flexible_mask import pressure,config as config_module
    from vpp2.full_mask.panel import file_hash
    root=tmp_path/'experiment'; source=root/'source'
    (source/'src').mkdir(parents=True)
    (source/'scripts').mkdir()
    (source/'src/model.py').write_text('unchanged = True\n')
    (source/'scripts/requirements.txt').write_text('torch==2.10.0\n')
    files={name:file_hash(source/name) for name in ('src/model.py','scripts/requirements.txt')}
    (root/'source_manifest.json').write_text(json.dumps(dict(source=str(source),files=files)))
    monkeypatch.setattr(pressure,'__file__',str(source/'src/vpp2/flexible_mask/pressure.py'))
    monkeypatch.setattr(config_module,'sources',lambda expected=files['src/model.py']:{'src/model.py':expected})
    return root,source,files


def test_publication_includes_unchanged_noncode_artifacts(tmp_path,monkeypatch):
    from vpp2.flexible_mask.pressure import published_sources
    root,source,files=publication_fixture(tmp_path,monkeypatch)
    assert published_sources({'root':str(root)})==files


@pytest.mark.parametrize('change',['alter_noncode','omit_code'])
def test_publication_rejects_changed_or_incomplete_artifacts(tmp_path,monkeypatch,change):
    from vpp2.flexible_mask.pressure import published_sources
    root,source,files=publication_fixture(tmp_path,monkeypatch)
    if change=='alter_noncode':
        (source/'scripts/requirements.txt').write_text('unreviewed dependency\n')
    else:
        files.pop('src/model.py')
        (root/'source_manifest.json').write_text(json.dumps(dict(source=str(source),files=files)))
    with pytest.raises(ValueError):published_sources({'root':str(root)})
