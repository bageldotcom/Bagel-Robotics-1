"""CPU checkpoint orchestration; distributed tensor numerics are GPU-preflight work."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from vpp2.flexible_mask import checkpoint
from vpp2.flexible_mask.controller import Masks
from vpp2.adaptive_mask.geometry import Geometry
from vpp2.flexible_mask.rng import rng_state
from vpp2.flexible_mask.preflight import fingerprint


class Engine:
    def __init__(self):
        self.module=torch.nn.Linear(2,2)
        self.optimizer=torch.optim.AdamW(self.module.parameters(),lr=.01)
        self.global_steps=1;self.loads=0
    def save_checkpoint(self,path,tag,client_state,save_latest):
        root=Path(path)/tag;root.mkdir(parents=True,exist_ok=True)
        torch.save(dict(model=self.module.state_dict(),optimizer=self.optimizer.state_dict(),
                        client=client_state,global_steps=self.global_steps),root/'rank0.pt')
    def load_checkpoint(self,path,tag,**kwargs):
        data=torch.load(Path(path)/tag/'rank0.pt',weights_only=False)
        self.module.load_state_dict(data['model']);self.optimizer.load_state_dict(data['optimizer'])
        self.global_steps=data['global_steps'];self.loads+=1
        return str(Path(path)/tag),data['client']


def setup(monkeypatch):
    monkeypatch.setattr(checkpoint.dist,'barrier',lambda:None)
    monkeypatch.setattr(checkpoint.dist,'all_gather_object',lambda target,value:target.__setitem__(0,value))
    g=Geometry(hidden=8,ffn=16,layers=2,heads=2,head_dim=8,pack=2,hidden_floor=6,ffn_floor=4,head_floor=1)
    masks=Masks(g,dict(hidden_scores=torch.arange(8.),ffn_scores=torch.arange(16.)[None],cross_scores=torch.arange(2.)[None]),
                dict(hidden_scores=6,ffn_scores=8,cross_scores=1))
    runtime=SimpleNamespace(rank=0,world=1,engine=Engine(),masks=masks,
        mask_optimizer=torch.optim.AdamW(masks.parameters(),lr=.01),recovery_export_sha256=None)
    runtime.engine.module(torch.ones(2,2)).square().mean().backward();runtime.engine.optimizer.step();runtime.engine.optimizer.zero_grad()
    sum(p.square().mean() for p in masks.parameters()).backward();runtime.mask_optimizer.step();runtime.mask_optimizer.zero_grad()
    return runtime


def test_checkpoint_restores_coupled_video_mask_optimizer_and_rng(tmp_path,monkeypatch):
    runtime=setup(monkeypatch)
    expected=fingerprint(dict(model=runtime.engine.module.state_dict(),optimizer=runtime.engine.optimizer.state_dict(),
        masks=runtime.masks.state_dict(),mask_optimizer=runtime.mask_optimizer.state_dict(),rng=rng_state()))
    checkpoint.save(runtime,tmp_path,dict(identity='identity',update=1,stage='mask'))
    assert (tmp_path/'checkpoints/latest').read_text().strip()=='update_000001'
    with torch.no_grad():
        for p in runtime.engine.module.parameters():p.add_(100)
        for p in runtime.masks.parameters():p.add_(100)
    torch.manual_seed(88)
    state=checkpoint.load(runtime,tmp_path,'identity','mask')
    assert state['cursor']==256
    actual=fingerprint(dict(model=runtime.engine.module.state_dict(),optimizer=runtime.engine.optimizer.state_dict(),
        masks=runtime.masks.state_dict(),mask_optimizer=runtime.mask_optimizer.state_dict(),rng=rng_state()))
    assert actual==expected


def test_changed_recovery_source_is_rejected_before_loading_weights(tmp_path,monkeypatch):
    runtime=setup(monkeypatch);runtime.recovery_export_sha256='original-export'
    checkpoint.save(runtime,tmp_path,dict(identity='identity',update=1,stage='recovery'))
    runtime.recovery_export_sha256='different-export'
    with pytest.raises(ValueError,match='initialization export'):checkpoint.load(runtime,tmp_path,'identity','recovery')
    assert runtime.engine.loads==0


def test_corrupt_partition_never_becomes_a_resumable_checkpoint(tmp_path,monkeypatch):
    runtime=setup(monkeypatch)
    checkpoint.save(runtime,tmp_path,dict(identity='identity',update=1,stage='mask'))
    p=tmp_path/'checkpoints/update_000001/rank0.pt'
    with p.open('ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError,match='Corrupt'):checkpoint.load(runtime,tmp_path,'identity','mask')
    assert runtime.engine.loads==0
