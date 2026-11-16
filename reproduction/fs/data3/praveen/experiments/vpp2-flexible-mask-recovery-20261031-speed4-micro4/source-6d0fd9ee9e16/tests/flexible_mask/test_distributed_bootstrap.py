"""Regression: ZeRO configuration must see the real distributed process group."""
import sys
from types import SimpleNamespace
import pytest
from vpp2.flexible_mask import runtime


def test_attach_registers_existing_group_before_zero_config(monkeypatch):
    calls=[];state={"initialized":False}
    monkeypatch.setattr(runtime.dist,"is_initialized",lambda:True)
    monkeypatch.setattr(runtime.dist,"get_backend",lambda:"nccl")
    monkeypatch.setattr(runtime.dist,"get_world_size",lambda:8)
    monkeypatch.setattr(runtime.dist,"get_rank",lambda:3)
    def attach(**kwargs):
        calls.append(kwargs);state["initialized"]=True
    def world():
        assert state["initialized"]
        return 8
    fake=SimpleNamespace(init_distributed=attach,comm=SimpleNamespace(get_world_size=world,get_rank=lambda:3))
    monkeypatch.setitem(sys.modules,"deepspeed",fake)
    runtime.attach_deepspeed_backend()
    assert calls==[dict(dist_backend="nccl",dist_init_required=False)]
    cfg=runtime.ds_config(dict(train=dict(global_batch=256,micro_batch=1)))
    assert cfg["train_batch_size"]==cfg["train_micro_batch_size_per_gpu"]*cfg["gradient_accumulation_steps"]*world()


@pytest.mark.parametrize("actual",[(1,0),(8,2)])
def test_mismatched_deepspeed_group_fails_closed(monkeypatch,actual):
    monkeypatch.setattr(runtime.dist,"is_initialized",lambda:True)
    monkeypatch.setattr(runtime.dist,"get_backend",lambda:"nccl")
    monkeypatch.setattr(runtime.dist,"get_world_size",lambda:8)
    monkeypatch.setattr(runtime.dist,"get_rank",lambda:3)
    fake=SimpleNamespace(init_distributed=lambda **kwargs:None,comm=SimpleNamespace(get_world_size=lambda:actual[0],get_rank=lambda:actual[1]))
    monkeypatch.setitem(sys.modules,"deepspeed",fake)
    with pytest.raises(RuntimeError,match="disagree"):runtime.attach_deepspeed_backend()
