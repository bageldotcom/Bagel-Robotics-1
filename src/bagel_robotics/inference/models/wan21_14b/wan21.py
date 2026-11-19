"""Shared video model with the strict compact inference loader."""
from bagel_robotics.models.backbone.wan21_14b.wan21 import Wan21VideoModel as SharedVideoModel
from bagel_robotics.inference.compact_loader import load_wan21_i2v_14b_components


class Wan21VideoModel(SharedVideoModel):
    component_loader = staticmethod(load_wan21_i2v_14b_components)
