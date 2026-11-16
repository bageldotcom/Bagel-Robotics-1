from types import SimpleNamespace
import copy
import pytest
import torch
from omegaconf import OmegaConf
from test_core import small
from vpp2.adaptive_mask import physical
from vpp2.adaptive_mask.controller import Controller
from vpp2.adaptive_mask.geometry import ParameterInventory
from vpp2.adaptive_mask.masking import install


def test_physical_audit_cannot_recompute_teacher_from_compact_policy():
    entries=[dict(id=7,owner_index=0),dict(id=8,owner_index=1)]
    model=SimpleNamespace(_adaptive_native_monitor=dict(teacher={7:torch.ones(3)}))
    physical.require_original_predictions(model,entries,0,2)
    with pytest.raises(ValueError,match='original native predictions'):
        physical.require_original_predictions(model,entries,1,2)
    saved=physical.require_original_predictions(model,entries,0,2)
    model._adaptive_native_monitor['teacher'][7].zero_()
    assert saved[7].eq(1).all()


def test_actual_sliced_module_is_used_and_original_reference_is_unchanged(monkeypatch):
    g,args,video,data=small();video.requires_grad_(False);install(video)
    model=SimpleNamespace(video_expert=video,device='cpu',
        _adaptive_native_monitor=dict(teacher={7:torch.ones(32,16)}))
    controller=Controller(g);selected=g.full();selected['hidden_indices']=[0,2,4,6,8,10]
    selected['ffn_indices'][0]=list(range(8));selected['cross_head_indices'][0]=[0]
    controller.set_selection(selected);inventory=ParameterInventory.from_model(video,g)
    before=copy.deepcopy(video.state_dict())
    monkeypatch.setattr(physical,'native_config',lambda config:OmegaConf.create(dict(model=dict(video_dit_config=args))))
    def evaluate(config,panel,reference,branch,masks,rank,world,parent,entries_override):
        assert branch.video is not reference.video_expert
        assert branch.video.hidden_dim==6 and branch.video.blocks[0].ffn[0].out_features==8
        assert branch.video.blocks[0].cross_attn.num_heads==1
        assert masks.forward_gates()==(None,None,None)
        assert reference._adaptive_native_monitor['teacher'][7].eq(1).all()
        assert parent=={'masked':'baseline'}
        return dict(passed=True)
    monkeypatch.setattr(physical,'native_evaluate',evaluate)
    result=physical.native_compact_audit({}, {}, model,controller,inventory,0,1,[dict(id=7)],{'masked':'baseline'})
    assert result['physical_video_parameters']==inventory.count(selected)
    assert result['original_predictions_reused'] and not result['masking_required']
    for name,value in before.items():torch.testing.assert_close(video.state_dict()[name],value,rtol=0,atol=0)
