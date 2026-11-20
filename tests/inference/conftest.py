"""Use the real XPolicyLab interface when installed, or a unit-test interface stub.

These CPU cases exercise Bagel state/history contracts. The installed benchmark
adapter and real checkpoint still require the documented integration checks.
"""
import importlib.util
import sys
from types import ModuleType


if "XPolicyLab" not in sys.modules and importlib.util.find_spec("XPolicyLab") is None:
    xpl = ModuleType("XPolicyLab")
    xpl.__path__ = []
    template = ModuleType("XPolicyLab.model_template")
    template.ModelTemplate = type("ModelTemplate", (), {})
    utils = ModuleType("XPolicyLab.utils")
    utils.__path__ = []
    process = ModuleType("XPolicyLab.utils.process_data")

    def requires_xpolicylab(_):
        raise RuntimeError("Install the pinned XPolicyLab checkout for integration")

    process.get_robot_action_dim_info = requires_xpolicylab
    sys.modules.update({"XPolicyLab": xpl, "XPolicyLab.model_template": template,
                        "XPolicyLab.utils": utils, "XPolicyLab.utils.process_data": process})
