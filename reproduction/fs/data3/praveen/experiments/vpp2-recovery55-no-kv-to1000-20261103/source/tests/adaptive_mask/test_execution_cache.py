import copy
from types import SimpleNamespace
import pytest
import torch
from vpp2.adaptive_mask.controller import Controller
from vpp2.adaptive_mask.geometry import Geometry
from vpp2.adaptive_mask.execution import configure_recomputation


def controller():
    g=Geometry(hidden=80,ffn=128,layers=3,heads=4,head_dim=8,pack=2,hidden_floor=76,ffn_floor=16,head_floor=1)
    c=Controller(g,100008);c.cache_hard_masks=True
    return c


def partial(c):
    s=c.selection();s['hidden_indices']=list(range(76));s['ffn_indices'][0]=list(range(0,128,2));s['cross_head_indices'][1]=[0,2,3]
    c.set_selection(s)


@pytest.mark.parametrize('pruned',[False,True])
def test_cache_preserves_repeated_optimizer_updates_and_invalidates_on_restore(pruned):
    a=controller()
    if pruned:partial(a)
    b=controller();b.load_state_dict(copy.deepcopy(a.state_dict()));b.cache_hard_masks=False
    oa=torch.optim.AdamW(a.parameters(),lr=.01);ob=torch.optim.AdamW(b.parameters(),lr=.01)
    saved=copy.deepcopy(a.state_dict())
    for update in range(3):
        oa.zero_grad();ob.zero_grad()
        for micro in range(4):
            ga=a.forward_gates();gb=b.forward_gates()
            for x,y in zip(ga,gb):assert torch.equal(x,y)
            for gates in (ga,gb):
                loss=sum((g*torch.linspace(-1,1,g.numel(),dtype=g.dtype).reshape_as(g)).sum() for g in gates)/4
                loss.backward()
        for x,y in zip(a.parameters(),b.parameters()):assert torch.equal(x.grad,y.grad)
        oa.step();ob.step()
        for x,y in zip(a.parameters(),b.parameters()):assert torch.equal(x,y)
    assert a._hard_mask_cache is not None
    a.load_state_dict(saved);assert a._hard_mask_cache is None
    b.load_state_dict(saved)
    for x,y in zip(a.forward_gates(),b.forward_gates()):assert torch.equal(x,y)
    assert not any('cache' in key for key in a.state_dict())


def test_selection_and_dtype_changes_invalidate_cache_and_binary_is_not_aliased():
    c=controller();c.forward_gates();old=c._hard_mask_cache
    c.forward_gates();assert c._hard_mask_cache is old
    binary=c.forward_gates(mode='binary');binary[0].zero_()
    assert c.forward_gates()[0].detach().eq(1).all()
    partial(c);assert c._hard_mask_cache is None
    g=c.forward_gates();assert g[0].detach().sum()==76
    c.double();g=c.forward_gates();assert all(x.dtype==torch.float64 for x in g)
    assert all(x.dtype==torch.float64 for x in c._hard_mask_cache[1])


@pytest.mark.parametrize('flag',[False,True])
def test_separate_attention_checkpoint_flag(flag):
    m=SimpleNamespace(video_expert=SimpleNamespace(blocks=[0,1]),action_expert=SimpleNamespace(),mot=SimpleNamespace(mot_checkpoint_mixed_attn=not flag))
    configure_recomputation(m,{'execution':{'mixed_attention_checkpointing':flag}})
    assert m.mot.mot_checkpoint_mixed_attn is flag


def test_invalid_attention_checkpoint_flag_fails_closed():
    m=SimpleNamespace(video_expert=SimpleNamespace(blocks=[0,1]),action_expert=SimpleNamespace(),mot=SimpleNamespace())
    with pytest.raises(ValueError):configure_recomputation(m,{'execution':{'mixed_attention_checkpointing':1}})


def test_execution_flags_reuse_existing_encoder_cache_but_semantic_changes_do_not():
    from vpp2.full_mask.conditioning_cache import conditioning_identity
    from vpp2.full_mask.panel import digest
    original={'teacher_bundle':'teacher_a','seed':100008,'execution':{'video_checkpoint_layers':[0,2],'action_checkpointing':False}}
    optimized=copy.deepcopy(original);optimized['execution'].update(cache_hard_masks=True,mixed_attention_checkpointing=False)
    assert conditioning_identity(optimized)==digest(original)
    changed=copy.deepcopy(optimized);changed['teacher_bundle']='teacher_b'
    assert conditioning_identity(changed)!=digest(original)
    changed=copy.deepcopy(optimized);changed['seed']=5
    assert conditioning_identity(changed)!=digest(original)
    changed=copy.deepcopy(optimized);changed['execution']['cache_hard_masks']=1
    with pytest.raises(ValueError):conditioning_identity(changed)
