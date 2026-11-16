"""Strict deployment bundles with no dependency on teacher weights or target caches."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import torch
from torch import nn
from safetensors.torch import load_model, save_model
from .action import FrozenPolicyCore
from .contracts import MemoryConfig, file_digest, write_json
from .memory import StructuredMemory
from .model import DistillationPolicy, QwenPrefix
from .native.action_dit import ActionDiT


def architecture(model):
    expert = model.action_core.runner.expert
    keys = ("hidden_dim", "action_dim", "ffn_dim", "text_dim", "freq_dim", "num_heads", "attn_head_dim")
    action = {k: getattr(expert, k) for k in keys}
    action["num_layers"] = len(expert.blocks)
    action["eps"] = expert.blocks[0].norm1.eps
    action["use_video_tokens"] = False
    return {"action": action, "memory": model.memory.cfg.to_dict(), "qwen_revision": model.encoder.revision,
            "inference": {"horizon":32, "execute":24, "steps":10, "shift":1., "seed":1}}


def save_architecture(model, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.encoder.model.config.save_pretrained(directory / "qwen_config")
    model.encoder.processor.save_pretrained(directory / "processor")
    write_json(directory / "architecture.json", architecture(model))


def construct(directory):
    directory = Path(directory)
    spec = json.loads((directory / "architecture.json").read_text())
    encoder = QwenPrefix.from_bundle(directory, spec["qwen_revision"])
    action = ActionDiT(**spec["action"])
    proprio = nn.Linear(16, spec["action"]["text_dim"])
    return DistillationPolicy(encoder, StructuredMemory(MemoryConfig(**spec["memory"]), checkpoint=False),
                              FrozenPolicyCore(action, proprio, checkpoint=False).to(dtype=torch.bfloat16))


def load_checkpoint_model(path):
    path = Path(path)
    metadata = json.loads((path / "progress.json").read_text())
    model = construct(path)
    state = torch.load(path / metadata["weights"], map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model, metadata


def export_checkpoint(checkpoint, output, data):
    from .dataset import Demonstrations
    model, metadata = load_checkpoint_model(checkpoint)
    dataset = Demonstrations(data, "train")
    if metadata["dataset_sha256"] != dataset.manifest["sha256"]:
        raise ValueError("Export dataset/statistics differ from training")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new deployment bundle directory")
    output.mkdir(parents=True)
    model.eval().requires_grad_(False)
    save_architecture(model, output)
    save_model(model, str(output / "model.safetensors"))
    shutil.copyfile(Path(data) / "dataset_stats.json", output / "dataset_stats.json")
    prompts = {}
    for episode in dataset.manifest["episodes"]:
        relative = episode["text"]
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(Path(data) / relative, destination)
        prompts[episode["prompt"]] = relative
    write_json(output / "text_index.json", prompts)
    files = {p.relative_to(output).as_posix(): file_digest(p) for p in output.rglob("*") if p.is_file()}
    write_json(output / "manifest.json", {"format":"vpp2-student-bundle-v1", "files":files,
               "teacher_identity":metadata["teacher_identity"], "dataset_sha256":metadata["dataset_sha256"],
               "source_checkpoint":str(Path(checkpoint).resolve()), "qwen_revision":model.encoder.revision})


def load_bundle(path, *, device="cuda", dtype=torch.bfloat16):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["format"] != "vpp2-student-bundle-v1":
        raise ValueError("Unknown deployment bundle")
    for name, checksum in manifest["files"].items():
        target = (path / name).resolve()
        target.relative_to(path.resolve())
        if file_digest(target) != checksum:
            raise ValueError(f"Deployment artifact changed: {name}")
    model = construct(path)
    missing, unexpected = load_model(model, str(path / "model.safetensors"), strict=True, device="cpu")
    if missing or unexpected:
        raise ValueError("Student checkpoint did not load strictly")
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return model, manifest
