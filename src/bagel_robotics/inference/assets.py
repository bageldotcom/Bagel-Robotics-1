"""Configured checkpoint identities and inference settings; no torch import."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .checkpoint_compat import COMPATIBILITY

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def load_manifest(path=None):
    """Load the default release identity, or an explicitly supplied model manifest."""
    return json.loads(Path(path or CONFIG_DIR / "artifacts.json").read_text())


MANIFEST = load_manifest()
FIXED_SETTINGS = json.loads((CONFIG_DIR / "settings.json").read_text())


def file_digest(path, algorithm="sha256", git_blob=False):
    path = Path(path)
    digest = hashlib.new(algorithm)
    if git_blob:
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(bundle, *, full_hash=True, manifest=None):
    manifest = MANIFEST if manifest is None else manifest
    bundle = Path(bundle).expanduser().resolve()
    for name, expected in manifest["files"].items():
        path = bundle / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint artifact: {path}")
        if path.stat().st_size != expected["size"]:
            raise ValueError(f"Checkpoint size mismatch: {name}")
        if full_hash or name not in {"action.pt", "video.pt"}:
            if file_digest(path) != expected["sha256"]:
                raise ValueError(f"Checkpoint SHA-256 mismatch: {name}")
    receipt = json.loads((bundle / "COMPLETE.json").read_text())
    if receipt.get("identity") != manifest["checkpoint_identity"] or receipt.get("update") != manifest["update"]:
        raise ValueError("Checkpoint identity does not match the configured model")
    architecture = json.loads((bundle / "compact_architecture.json").read_text())
    if architecture.get("schema") != "vpp2-compact-channels-v1" or architecture.get("parameters") != manifest["video_parameters"]:
        raise ValueError("Unexpected compact architecture")
    return {"checkpoint_identity": receipt["identity"], "update": manifest["update"],
            "files": len(manifest["files"]), "full_hash": full_hash}


def verify_wan(wan_root, *, full_hash=True, manifest=None):
    manifest = MANIFEST if manifest is None else manifest
    wan_root = Path(wan_root).expanduser().resolve()
    prefix = "checkpoints/Wan2.1-I2V-14B-480P/"
    for entry in manifest["wan"]["files"]:
        path = wan_root / entry["rfilename"].removeprefix(prefix)
        if not path.is_file():
            raise FileNotFoundError(f"Missing encoder asset: {path}")
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"Encoder asset size mismatch: {path.name}")
        if full_hash:
            lfs = entry.get("lfs")
            actual = file_digest(path, "sha256" if lfs else "sha1", git_blob=not bool(lfs))
            expected = lfs["sha256"] if lfs else entry["blobId"]
            if actual != expected:
                raise ValueError(f"Encoder asset checksum mismatch: {path.name}")
    return {"repo": manifest["wan"]["repo"], "revision": manifest["wan"]["revision"],
            "files": len(manifest["wan"]["files"]), "full_hash": full_hash}


def validate_settings(model_cfg):
    cfg = dict(model_cfg)
    for key, expected in FIXED_SETTINGS.items():
        actual = cfg.get(key, expected)
        if actual != expected:
            raise ValueError(f"Released policy requires {key}={expected!r}; got {actual!r}")
        cfg[key] = expected
    seed = cfg.get("seed", 1)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer supplied by the evaluator")
    cfg["seed"] = seed
    cfg.setdefault("ckpt_name", "default")
    cfg.setdefault("video_parameters", MANIFEST["video_parameters"])
    return cfg


def prepare_model_config(model_cfg, policy_dir=None):
    from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root

    policy_dir = Path(policy_dir or Path.cwd())
    cfg = validate_settings(model_cfg)
    manifest = load_manifest(cfg["artifact_manifest"]) if cfg.get("artifact_manifest") else MANIFEST
    explicit = next((os.environ[name] for name in COMPATIBILITY["environment_aliases"]["bundle"]
                     if os.environ.get(name)), None) or cfg.get("bundle_path")
    lookup = dict(cfg)
    if explicit:
        lookup["model_path"] = explicit
    bundle = resolve_checkpoint_root(lookup, policy_dir / "checkpoints", policy_dir=policy_dir)
    wan_root = next((os.environ[name] for name in COMPATIBILITY["environment_aliases"]["wan_root"]
                     if os.environ.get(name)), None) or cfg.get("wan_model_dir")
    if not wan_root:
        wan_root = policy_dir / "checkpoints/Wan2.1-I2V-14B-480P"
    verification = {"checkpoint": verify_bundle(bundle, manifest=manifest),
                    "encoders": verify_wan(wan_root, manifest=manifest)}
    print("[Bagel-Robotics-1] " + json.dumps(verification), flush=True)
    cfg.update(checkpoint_path=str(bundle / "action.pt"),
               video_checkpoint_path=str(bundle / "video.pt"),
               dataset_stats_path=str(bundle / "dataset_stats.json"),
               wan_model_dir=str(Path(wan_root).expanduser().resolve()),
               video_parameters=manifest["video_parameters"])
    return cfg
