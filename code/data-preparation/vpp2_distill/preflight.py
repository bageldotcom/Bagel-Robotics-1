"""Offline checks for staged data and weights; never connects to or starts a teacher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch
import yaml

from .checkpoints import checkpoint_settings, require_checkpoint_space
from .contracts import EE16Normalizer, UPSTREAM_COMMIT, file_digest, validate_native_config, write_json
from .prepare import require_upstream


def check(config, *, full_data=False, load_models=False):
    runtime, model = config["runtime"], config["model"]
    report = {"passed": False, "checks": {}, "pending": []}
    checks = report["checks"]
    if config.get("teacher_mode") != "online":
        raise ValueError("This preflight checks the frozen-online experiment")
    require_upstream(runtime["vpp2_root"])
    checks["native_revision"] = UPSTREAM_COMMIT
    for name in ("student_python", "teacher_python"):
        if not Path(runtime[name]).is_file():
            raise FileNotFoundError(runtime[name])
    for name in ("student", "teacher"):
        code = "import json,torch,transformers,accelerate; versions=dict(torch=torch.__version__,transformers=transformers.__version__,accelerate=accelerate.__version__); "
        if name == "student":
            code += "import deepspeed; from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer; versions['deepspeed']=deepspeed.__version__; "
        result = subprocess.check_output([runtime[name+"_python"], "-c", code+"print(json.dumps(versions))"], text=True)
        checks[name+"_environment"] = json.loads(result.strip().splitlines()[-1])
    if runtime.get("build_bin"):
        ninja = Path(runtime["build_bin"])/"ninja"
        checks["build_tools"] = {"ninja":subprocess.check_output([str(ninja), "--version"], text=True).strip(),
                                 "path":str(ninja), "sha256":file_digest(ninja)}
        if not shutil.which("gcc") or not shutil.which("g++"):
            raise RuntimeError("C/C++ compiler is missing")
    split = Path(runtime["prepared_split"])
    bootstrap = json.loads((split / "bootstrap_complete.json").read_text())
    for path, expected in bootstrap["files"].items():
        if file_digest(split / path) != expected:
            raise ValueError(f"Prepared split/text checksum mismatch: {path}")
    checks["split"] = bootstrap["summary"]
    normalizer = EE16Normalizer(split / "dataset_stats.json")
    from .native.checkpoint_compat import action_config
    action = torch.load(model["action_checkpoint"], map_location="cpu", mmap=True, weights_only=True)
    validate_native_config(action_config(action))
    if action["step"] != 100000:
        raise ValueError("Expected joint Action2B step 100000")
    video = torch.load(runtime["video_checkpoint"], map_location="cpu", mmap=True, weights_only=True)
    if video.get("step") != action["step"] or "dit" not in video:
        raise ValueError("Native Action2B/video checkpoint step or format mismatch")
    checks["paired_native_step"] = action["step"]
    del action, video
    wan = Path(runtime["wan"])
    tokenizer = wan / "google/umt5-xxl"
    if not (tokenizer/"tokenizer_config.json").is_file() or not (tokenizer/"spiece.model").is_file():
        raise FileNotFoundError(f"Native local UMT5 tokenizer is incomplete: {tokenizer}")
    assets = {"action":Path(model["action_checkpoint"]), "video":Path(runtime["video_checkpoint"]),
              **{name:wan/name for name in ("Wan2.1_VAE.pth", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                                           "models_t5_umt5-xxl-enc-bf16.pth")}}
    checks["native_assets"] = {name:{"path":str(path), "bytes":path.stat().st_size, "sha256":file_digest(path)}
                              for name,path in assets.items()}
    if checks["native_assets"]["models_t5_umt5-xxl-enc-bf16.pth"]["sha256"] != bootstrap["identity"]["text_encoder_sha256"]:
        raise ValueError("Prepared contexts came from another text encoder")
    if normalizer.identity != bootstrap["identity"]["stats_sha256"]:
        raise ValueError("Prepared normalization differs from the pinned native statistics")
    qwen = json.loads(Path(runtime["qwen_report"]).read_text())
    if Path(qwen["path"]).resolve() != Path(model["qwen_model"]).resolve() or qwen["revision"] != model["qwen_revision"]:
        raise ValueError("Qwen model differs from the pinned staged snapshot")
    for name, meta in qwen["files"].items():
        path = Path(qwen["path"])/name
        if path.stat().st_size != meta["bytes"] or file_digest(path) != meta["sha256"]:
            raise ValueError(f"Qwen snapshot checksum mismatch: {name}")
    checks["qwen_revision"] = qwen["revision"]
    data_root = Path(config["data_root"])
    if (data_root / "manifest.json").is_file():
        from .dataset import Demonstrations
        train, val = Demonstrations(data_root), Demonstrations(data_root, "validation")
        expected_splits = {name:file_digest(split/name) for name in ("train.csv", "val.csv")}
        if (train.manifest.get("split_sha256") != expected_splits or
                train.manifest.get("source_info_sha256") != bootstrap["identity"]["source_info_sha256"] or
                train.normalizer.identity != normalizer.identity or
                (len(train.episodes), len(val.episodes), len(train)+len(val)) !=
                (bootstrap["summary"]["train"], bootstrap["summary"]["validation"], bootstrap["summary"]["frames"])):
            raise ValueError("Decoded dataset differs from the verified official source/split")
        for dataset in (train, val):
            # Audit array metadata/text for every episode, with optional full byte checks.
            for i, episode in enumerate(dataset.episodes):
                dataset._episode(i)
                if full_data:
                    for name, expected in episode["files"].items():
                        if file_digest(data_root/episode["path"]/name) != expected:
                            raise ValueError(f"Prepared episode checksum mismatch: {episode['id']}/{name}")
        checks["data"] = {"manifest":train.manifest["sha256"], "train_episodes":len(train.episodes),
                          "validation_episodes":len(val.episodes), "train_windows":len(train),
                          "validation_windows":len(val), "full_byte_audit":full_data,
                          "text_encoding":train.manifest.get("text_encoding")}
    else:
        completed = sum(1 for _ in (data_root / "episodes").glob("*/complete.json"))
        report["pending"].append(f"Decoded-frame preparation: {completed}/3500 episodes complete; manifest not yet published")
    if load_models:
        from .model import QwenPrefix
        from .action import FrozenPolicyCore
        encoder = QwenPrefix.pretrained(model["qwen_model"], model["qwen_revision"])
        if encoder.model.config.text_config.hidden_size != model["memory"]["qwen_dim"]:
            raise ValueError("Qwen hidden width differs from the configured memory decoder")
        checks["qwen_cpu_strict_load"] = {"parameters":sum(p.numel() for p in encoder.parameters()), "passed":True}
        from PIL import Image
        files = sorted((data_root/"episodes").glob("*/complete.json"))
        if files:
            frames = np.load(files[0].parent/"frames.npy", mmap_mode="r")
            image = Image.fromarray(np.array(frames[0]).transpose(1,2,0))
            content = [{"type":"image", "image":image} for _ in range(9)]
            content.append({"type":"text", "text":"Move the object into the box."})
            inputs = encoder.processor.apply_chat_template([[{"role":"user", "content":content}]], tokenize=True,
                        add_generation_prompt=True, padding=True, return_tensors="pt", return_dict=True)
            checks["qwen_real_image_processor"] = {name:list(value.shape) for name,value in inputs.items() if hasattr(value,"shape")}
        del encoder
        core = FrozenPolicyCore.from_checkpoint(model["action_checkpoint"])
        checks["action_cpu_strict_load"] = {"parameters":sum(p.numel() for p in core.parameters()), "passed":True}
        del core
    experiment = Path(runtime["experiment"])
    settings = checkpoint_settings(config)
    require_checkpoint_space(experiment, settings)
    checks["output_storage"] = {"root":str(experiment), "free_gib":shutil.disk_usage(experiment).free/2**30,
                               "checkpointing":settings}
    students, teachers = int(runtime["student_processes"]), int(runtime["teacher_processes"])
    if students < 1 or teachers < 1 or 32 % (students*int(config["microbatch"])):
        raise ValueError("Invalid teacher/student process layout for global batch 32")
    checks["gpu_layout"] = {"teacher":teachers, "student":students, "total":students+teachers}
    report["passed"] = not report["pending"]
    report["gpu_checks"] = {"status":"not_run", "required":["native 10-step BF16 Action2B parity",
        "teacher-to-student-runtime Action2B/proprio parity on identical native K/V",
        "full-model forward/backward and peak GPU memory", "ZeRO-2 save/resume in warmup and main",
        "prepared UMT5 contexts exactly match native GPU encoding when required"]}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/robodojo_qwen4b_action2b_online_node.yaml")
    parser.add_argument("--report", required=True)
    parser.add_argument("--full-data", action="store_true")
    parser.add_argument("--load-models", action="store_true", help="Strictly load Qwen4B and Action2B on CPU")
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = check(yaml.safe_load(Path(args.config).read_text()), full_data=args.full_data, load_models=args.load_models)
    write_json(args.report, report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
