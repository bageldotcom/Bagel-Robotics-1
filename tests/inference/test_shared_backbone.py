"""Shared model mathematics must retain separate training/inference loaders."""
import pytest

from bagel_robotics.inference.checkpoint_compat import canonicalize_config
from bagel_robotics.inference.compact_loader import load_wan21_i2v_14b_components as strict_loader
from bagel_robotics.inference.models.wan21_14b.vpp2 import VPP2
from bagel_robotics.inference.models.wan21_14b.wan21 import Wan21VideoModel
from bagel_robotics.models.backbone.wan21_14b.helpers.loader import load_wan21_i2v_14b_components as training_loader
from bagel_robotics.models.backbone.wan21_14b.vpp2 import VPP2 as SharedVPP2
from bagel_robotics.models.backbone.wan21_14b.wan21 import Wan21VideoModel as SharedVideoModel


def test_shared_math_keeps_loader_bindings_separate():
    assert VPP2.component_loader is Wan21VideoModel.component_loader is strict_loader
    assert SharedVPP2.component_loader is SharedVideoModel.component_loader is training_loader
    assert VPP2.infer_action is SharedVPP2.infer_action
    assert Wan21VideoModel.infer is SharedVideoModel.infer


def test_inference_factory_uses_strict_subclass_binding(monkeypatch):
    class StopAfterLoader(Exception):
        pass

    def probe(**kwargs):
        assert kwargs["device"] == "cpu"
        assert kwargs["dit_config"]["text_dim"] > 0
        raise StopAfterLoader

    monkeypatch.setattr(VPP2, "component_loader", staticmethod(probe))
    with pytest.raises(StopAfterLoader):
        VPP2.from_wan21_14b_pretrained(device="cpu")
    assert SharedVPP2.component_loader is training_loader


@pytest.mark.parametrize("suffix, expected", [
    ("action_dit.ActionDiT", "bagel_robotics.models.backbone.wan21_14b.action_dit.ActionDiT"),
    ("vpp2.VPP2", "bagel_robotics.inference.models.wan21_14b.vpp2.VPP2"),
    ("helpers.loader.load_wan21_i2v_14b_components", "bagel_robotics.inference.compact_loader.load_wan21_i2v_14b_components"),
])
def test_checkpoint_targets_select_shared_leaves_and_strict_factories(suffix, expected):
    result = canonicalize_config({"_target_": "vpp2.models.wan21_14b." + suffix})
    assert result["_target_"] == expected
