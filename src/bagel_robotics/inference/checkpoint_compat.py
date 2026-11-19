"""Load serialized checkpoint configuration into the canonical package.

Serialized identifiers are normalized without importing another model package
or changing tensor keys.
"""

import json
from pathlib import Path

COMPATIBILITY = json.loads((Path(__file__).resolve().parent / "configs/compatibility.json").read_text())
ACTION_FORMAT = COMPATIBILITY["action_format"]
NAMESPACE_ALIASES = tuple(COMPATIBILITY["namespace_aliases"])


def canonicalize_config(value):
    """Convert Hydra targets recursively, preserving paths and other values."""
    if isinstance(value, list):
        return [canonicalize_config(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: canonicalize_config(item) for key, item in value.items()}
    target = result.get("_target_")
    if isinstance(target, str):
        for prefix in NAMESPACE_ALIASES:
            if target.startswith(prefix):
                target = "vpp2." + target[len(prefix) :]
                break
        if target.startswith("vpp2."):
            target = ".".join(COMPATIBILITY["target_parts"].get(part, part) for part in target.split("."))
            target = "bagel_robotics.inference." + target[len("vpp2.") :]
        model_prefix = "bagel_robotics.inference.models.wan21_14b."
        if target.startswith(model_prefix):
            suffix = target[len(model_prefix):]
            if not suffix.startswith(("vpp2.", "wan21.")):
                if suffix.startswith("helpers.loader."):
                    target = "bagel_robotics.inference.compact_loader." + suffix[len("helpers.loader."):]
                else:
                    target = "bagel_robotics.models.backbone.wan21_14b." + suffix
        result["_target_"] = target
    return result


def action_config(payload):
    if payload.get("format") not in {ACTION_FORMAT, *COMPATIBILITY["accepted_action_formats"]}:
        raise ValueError("Expected an exported VPP2 action checkpoint")
    return canonicalize_config(payload["resolved_config"])
