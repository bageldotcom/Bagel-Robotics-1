"""Shared world-action model with the strict compact inference loader."""
from bagel_robotics.models.backbone.wan21_14b.vpp2 import VPP2 as SharedVPP2
from bagel_robotics.inference.compact_loader import load_wan21_i2v_14b_components


class VPP2(SharedVPP2):
    component_loader = staticmethod(load_wan21_i2v_14b_components)
