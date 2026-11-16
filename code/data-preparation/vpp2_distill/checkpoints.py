"""Disk bounds and retention restricted to completed checkpoints of one run."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import shutil

from .contracts import digest, file_digest

CHECKPOINT_FORMAT = "vpp2-distill-checkpoint-v1"
OWNERSHIP = ("signature", "stage", "refresh_round", "teacher_identity", "dataset_sha256", "world_size", "total_steps")


def source_identity():
    """Fingerprint the actual training/target code, including uncommitted files."""
    root = Path(__file__).resolve().parent
    names = ["train.py", "model.py", "memory.py", "action.py", "losses.py", "contracts.py", "dataset.py",
             "batches.py", "online.py", "teacher.py", "teacher_service.py", "bundle.py", "checkpoints.py"]
    paths = [root/name for name in names] + sorted((root/"native").rglob("*.py"))
    files = {p.relative_to(root).as_posix():file_digest(p) for p in paths}
    return {"sha256":digest(files), "files":files}


def recipe_config(config):
    """Transport addresses and storage policy do not change the training recipe."""
    result = copy.deepcopy(config)
    for key in ("output", "checkpointing", "runtime"):
        result.pop(key, None)
    if "online_teacher" in result:
        for key in ("urls", "timeout"):
            result["online_teacher"].pop(key, None)
    return result


def checkpoint_settings(config):
    settings = {"keep_last": 3, "every_steps": 2000, "min_free_gib": 0, "expected_checkpoint_gib": 0,
                **config.get("checkpointing", {})}
    if settings["keep_last"] < 1 or settings["every_steps"] < 1:
        raise ValueError("Checkpoint retention and interval must be positive")
    if settings["min_free_gib"] < 0 or settings["expected_checkpoint_gib"] < 0:
        raise ValueError("Checkpoint disk bounds must be non-negative")
    return settings


def require_checkpoint_space(output, settings):
    free = shutil.disk_usage(output).free
    required = int((settings["min_free_gib"] + settings["expected_checkpoint_gib"]) * 2**30)
    if free < required:
        raise OSError(f"Checkpoint needs {required / 2**30:.1f} GiB free including reserve; found {free / 2**30:.1f}")


def prune_checkpoints(output, current, keep_last):
    """Only prune this stage's fully published, owned checkpoints, after a save."""
    output, current = Path(output), Path(current)
    if output.is_symlink() or current.is_symlink() or current.parent != output:
        raise ValueError("Checkpoint retention requires real directories within this run")
    owner = json.loads((current / "progress.json").read_text())
    if owner.get("checkpoint_format") != CHECKPOINT_FORMAT or keep_last < 1:
        raise ValueError("Checkpoint retention requires a recognized completed checkpoint")
    eligible = []
    for path in output.iterdir():
        if path.is_symlink() or not path.is_dir() or not re.fullmatch(r"step_[0-9]{7,}", path.name):
            continue
        if not (path / "progress.json").is_file() or any(p.is_symlink() for p in path.rglob("*")):
            continue
        try:
            meta = json.loads((path / "progress.json").read_text())
        except (ValueError, OSError):
            continue
        if meta.get("checkpoint_format") != CHECKPOINT_FORMAT or any(meta.get(k) != owner.get(k) for k in OWNERSHIP):
            continue
        weights = meta.get("weights", "")
        if not weights or Path(weights).name != weights or not (path / weights).is_file() or not (path / "architecture.json").is_file():
            continue
        if not isinstance(meta.get("step"), int) or path.name != f"step_{meta['step']:07d}":
            continue
        eligible.append((meta["step"], path))
    eligible.sort(reverse=True)
    removed = []
    for _, path in eligible[keep_last:]:
        if path == current:
            continue
        shutil.rmtree(path)
        removed.append(str(path))
    return removed
