"""XPolicyLab entry point for the canonical Bagel policy implementation."""
from pathlib import Path

from bagel_robotics.inference.robodojo import Model as BagelModel


class Model(BagelModel):
    def __init__(self, model_cfg):
        super().__init__(model_cfg, policy_dir=Path(__file__).resolve().parent)
