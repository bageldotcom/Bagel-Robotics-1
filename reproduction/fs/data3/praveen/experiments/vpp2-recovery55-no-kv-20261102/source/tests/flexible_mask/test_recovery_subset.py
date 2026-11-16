"""Recovery dataset phases preserve optimizer history and isolate the sample cursor."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from vpp2.flexible_mask import recovery_subset as phase
from vpp2.flexible_mask import checkpoint
from vpp2.flexible_mask.data import stream,positions
from vpp2.flexible_mask.train import learning_rate_fraction
from vpp2.flexible_mask.preflight import fingerprint
from vpp2.flexible_mask.rng import rng_state


def config():
    return {'root':'/new','train':{'global_batch':256,'micro_batch':4},
        'recovery':{'updates':2344,'learning_rate':5e-6,'warmup_updates':250,'final_lr_fraction':.1},
        'wandb':{'name':'new','group':'new'},
        'recovery_subset':dict(schema=phase.SCHEMA,sampler='original_curated_task_balanced',
            windows=30000,updates=2344,parent_root='/old',parent_update=55,parent_identity='old-id',
            parent_tag='update_000055',parent_config_sha256='x',parent_sources_sha256='x',
            parent_inputs_sha256='x',parent_receipt_sha256='x',parent_checkpoint_identity='checkpoint',
            parent_export_receipt_sha256='x',parent_selected_sha256='x',export_manifest_sha256='x',
            export_path='/old/export')}


def test_phase_budget_is_additional_and_noise_positions_continue():
    c=config();phase.validate_settings(c)
    assert phase.endpoint(c)==2399
    all_positions=[]
    for rank in range(8):
        for batch in positions(56,rank,8,4):all_positions.extend(batch)
    assert sorted(phase.sample_position(c,p) for p in all_positions)==list(range(256))
    assert sorted(all_positions)==list(range(55*256,56*256))
    assert phase.sample_position(c,2399*256-1)==2344*256-1
    with pytest.raises(ValueError,match='precedes'):phase.sample_position(c,55*256-1)
    assert learning_rate_fraction(56,2399,250)==learning_rate_fraction(56,6000,250)
    assert learning_rate_fraction(2399,2399,250)==pytest.approx(.1)


def test_dataset_cursor_survives_resume_without_becoming_global_cursor():
    c=config();state={'update':102,**phase.state_fields(c,102)}
    phase.check_state(c,state)
    assert state['recovery_subset_cursor']==47*256
    bad=dict(state,recovery_subset_cursor=102*256)
    with pytest.raises(ValueError,match='cursor'):phase.check_state(c,bad)
    with pytest.raises(ValueError,match='outside'):phase.state_fields(c,2400)
    with pytest.raises(ValueError,match='outside'):phase.state_fields(c,54)


def test_unrelated_recipe_changes_and_mixed_phase_are_rejected():
    c=config();old=copy.deepcopy(c);old.pop('recovery_subset');old['root']='/old';old['recovery']['updates']=6000
    phase.validate_recipe(c,old)
    for section,key,value in [('train','global_batch',128),('recovery','learning_rate',1e-5)]:
        changed=copy.deepcopy(c);changed[section][key]=value
        with pytest.raises(ValueError,match='unrelated'):phase.validate_recipe(changed,old)
    bad=copy.deepcopy(c);bad['resume_from']={'anything':True}
    with pytest.raises(ValueError,match='combine'):phase.validate_settings(bad)
    bad=copy.deepcopy(c);bad['root']='/old'
    with pytest.raises(ValueError,match='separate'):phase.validate_settings(bad)


def test_curated_stream_never_expands_episode_frames(monkeypatch):
    from vpp2.flexible_mask import data
    rows=[dict(id=i,task='task'+str(i//3),episode=i//3,length=1000,frame=i*7) for i in range(6)]
    monkeypatch.setattr(data,'validate_panel',lambda panel:(rows,[]))
    selected=stream({},100008,recovery=True,curated_recovery=True)
    full=stream({},100008,recovery=True)
    assert sum(selected.sizes.values())==6 and sum(full.sizes.values())==1938
    expected={(r['episode'],r['frame']) for r in rows}
    for cycle in range(3):
        assert {(selected.at(i)['episode'],selected.at(i)['frame']) for i in range(cycle*6,(cycle+1)*6)}==expected
    resumed=stream({},100008,recovery=True,curated_recovery=True)
    assert [resumed.at(i)['id'] for i in range(7,37)]==[selected.at(i)['id'] for i in range(7,37)]


class Engine:
    def __init__(self):
        self.module=torch.nn.Linear(2,2)
        self.optimizer=torch.optim.AdamW(self.module.parameters(),lr=.01)
        self.global_steps=55
        self.module(torch.ones(2,2)).square().mean().backward()
        self.optimizer.step();self.optimizer.zero_grad()
    def save_checkpoint(self,path,tag,client_state,save_latest):
        root=Path(path)/tag;root.mkdir(parents=True,exist_ok=True)
        torch.save(dict(model=self.module.state_dict(),optimizer=self.optimizer.state_dict(),
            client=client_state,steps=self.global_steps),root/'rank0.pt')
    def load_checkpoint(self,path,tag,**kwargs):
        d=torch.load(Path(path)/tag/'rank0.pt',weights_only=False)
        self.module.load_state_dict(d['model']);self.optimizer.load_state_dict(d['optimizer'])
        self.global_steps=d['steps']
        return str(Path(path)/tag),d['client']


def test_parent_model_optimizer_rng_preserved_then_new_phase_checkpoint_roundtrip(tmp_path,monkeypatch):
    monkeypatch.setattr(checkpoint.dist,'barrier',lambda:None)
    monkeypatch.setattr(checkpoint.dist,'all_gather_object',lambda target,value:target.__setitem__(0,value))
    rt=SimpleNamespace(rank=0,world=1,engine=Engine(),masks=None,mask_optimizer=None,recovery_export_sha256='export')
    parent=tmp_path/'old';child=tmp_path/'new';c=config();c['root']=str(child)
    c['recovery_subset']['parent_root']=str(parent)
    state=dict(identity='old-id',update=55,stage='recovery',history=[{'parent':1}],best=[{'update':30}])
    path=checkpoint.save(rt,parent/'recovery',state)
    receipt=json.loads((path/'COMPLETE.json').read_text())
    c['recovery_subset']['parent_checkpoint_identity']=receipt['checkpoint_identity']
    monkeypatch.setattr(phase,'validate_parent',lambda config:({'root':str(parent)},receipt))
    from vpp2.flexible_mask import config as cfg
    monkeypatch.setattr(cfg,'identity',lambda c:'new-id')
    expected=fingerprint({'model':rt.engine.module.state_dict(),'optimizer':rt.engine.optimizer.state_dict(),'rng':rng_state()})
    with torch.no_grad():
        for p in rt.engine.module.parameters():p.add_(100)
    torch.manual_seed(77)
    loaded=phase.load_parent(rt,c)
    assert fingerprint({'model':rt.engine.module.state_dict(),'optimizer':rt.engine.optimizer.state_dict(),'rng':rng_state()})==expected
    assert loaded['identity']=='new-id' and loaded['update']==55 and loaded['cursor']==55*256
    assert loaded['recovery_subset_cursor']==0 and loaded['history']==[] and loaded['best']==[]
    assert loaded['parent_history']==[{'parent':1}]
    # A resumed child must retain the phase cursor as well as the global counter.
    rt.engine.global_steps=56
    loaded.update(update=56,**phase.state_fields(c,56))
    checkpoint.save(rt,child/'recovery',loaded)
    restored=checkpoint.load(rt,child/'recovery','new-id','recovery')
    phase.check_state(c,restored)
    assert restored['cursor']==56*256 and restored['recovery_subset_cursor']==256
