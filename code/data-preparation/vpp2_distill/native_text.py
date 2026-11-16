"""Create a new demonstration manifest with native GPU-encoded text contexts."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

import torch
import yaml

from .contracts import digest, file_digest, write_json
from .dataset import Demonstrations
from .teacher import Teacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    source, output = Path(config["data_root"]).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite a text overlay: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("Native text preparation requires an allocated GPU")
    data = Demonstrations(source)
    runtime, model = config["runtime"], config["model"]
    teacher_args = argparse.Namespace(vpp2_root=runtime["vpp2_root"], action=model["action_checkpoint"],
        video=runtime["video_checkpoint"], wan=runtime["wan"], stats=str(source/"dataset_stats.json"), device="cuda:0")
    teacher = Teacher(teacher_args, data.normalizer.identity)
    output.mkdir(parents=True)
    (output/"episodes").symlink_to(source/"episodes", target_is_directory=True)
    shutil.copy2(source/"dataset_stats.json", output/"dataset_stats.json")
    manifest = copy.deepcopy(data.manifest)
    contexts = {}
    for episode in manifest["episodes"]:
        prompt = episode["prompt"]
        if prompt not in contexts:
            with torch.no_grad():
                context, mask = teacher.model.encode_prompt([prompt])
            context, mask = context[0].cpu().contiguous(), mask[0].cpu().contiguous()
            if context.shape != (128,4096) or context.dtype != torch.bfloat16 or not torch.isfinite(context).all() or not mask.all():
                raise ValueError("Unexpected native prompt encoding")
            path = output/episode["text"]
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(prompt=prompt, context=context, mask=mask), path)
            contexts[prompt] = file_digest(path)
        episode["text_sha256"] = contexts[prompt]
    manifest["text_encoding"] = dict(device=torch.cuda.get_device_name(), torch=str(torch.__version__),
        source_manifest=data.manifest["sha256"], method="native VPP2 encode_prompt; batch size 1",
        teacher_identity=teacher.identity, prompts=len(contexts))
    manifest["sha256"] = digest({k:v for k,v in manifest.items() if k != "sha256"})
    write_json(output/"manifest.json", manifest)
    audit = teacher.audit_contexts(Demonstrations(output))
    report = dict(passed=audit["all_bitwise_equal"], source=str(source), output=str(output),
        manifest=manifest["sha256"], text_encoding=manifest["text_encoding"], audit=audit)
    write_json(args.report, report)
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Saved contexts differ from fresh native prompt encoding")


if __name__ == "__main__":
    main()
