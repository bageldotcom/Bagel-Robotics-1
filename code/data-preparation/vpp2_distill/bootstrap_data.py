"""Reconstruct distillation metadata from the pinned official split; encode text on CPU."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import torch

from .contracts import digest, file_digest, write_json
from .prepare import require_upstream


def bootstrap(args):
    upstream = require_upstream(args.vpp2_root)
    from vpp2.convert_robodojo import load_episodes
    from vpp2.data import DEFAULT_PROMPT, text_cache
    source, output, wan = Path(args.source).resolve(), Path(args.output).resolve(), Path(args.wan).resolve()
    tasks, holdout = upstream / "configs/robodojo/source_tasks.csv", upstream / "configs/robodojo/holdout.json"
    episodes = load_episodes(source, tasks)
    split = json.loads(holdout.read_text())
    heldout = {int(row["episode_index"]): row for row in split["held_out"]}
    by_id = {e["index"]: e for e in episodes}
    for index, row in heldout.items():
        if by_id[index]["task"]["task_name"] != row["task_name"] or by_id[index]["length"] != row["num_frames"]:
            raise ValueError("Source dataset differs from the published holdout")
    stats = json.loads(Path(args.stats).read_text())
    reference_stats = json.loads((upstream / "configs/robodojo/normalization_ee16.json").read_text())
    if digest(stats) != digest(reference_stats):
        raise ValueError("Checkpoint statistics differ from the pinned reference statistics")
    provenance = dict(format="vpp2-distill-official-split-v1", upstream=str(upstream),
                      source_info_sha256=file_digest(source / "meta/info.json"),
                      tasks_sha256=file_digest(tasks), holdout_sha256=file_digest(holdout),
                      stats_sha256=digest(stats), text_device=args.device,
                      text_encoder_sha256=file_digest(wan / "models_t5_umt5-xxl-enc-bf16.pth"))
    report_path = output / "bootstrap_complete.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["identity"] != provenance or any(file_digest(output / name) != checksum for name, checksum in report["files"].items()):
            raise ValueError("Completed metadata/text contents changed")
        print(json.dumps(report["summary"]), flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "bootstrap_identity.json"
    if marker.exists() and json.loads(marker.read_text()) != provenance:
        raise ValueError("Existing metadata belongs to another preparation")
    write_json(marker, provenance)
    rows, prompts = [], {}
    for episode in episodes:
        task = episode["task"]
        prompt = DEFAULT_PROMPT.format(task=task["instruction"])
        if prompt not in prompts:
            prompts[prompt] = f"text_embeds/prompt_{len(prompts):05d}.pt"
        rows.append(dict(episode_index=episode["index"], task_name=task["task_name"],
                         dimension=task["dimension"], fps=25, num_frames=episode["length"],
                         trim_start=0, trim_end=episode["length"], high_level_instruction=task["instruction"],
                         training_prompt=prompt, text_embedding_path=prompts[prompt]))
    train = [row for row in rows if row["episode_index"] not in heldout]
    val = [row for row in rows if row["episode_index"] in heldout]
    if (len(train), len(val)) != (split["train_episodes"], split["val_episodes"]):
        raise ValueError("Published split counts differ")
    for name, items in (("all.csv", rows), ("train.csv", train), ("val.csv", val)):
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(items)
    shutil.copyfile(args.stats, output / "dataset_stats.json")
    write_json(output / "prompts.json", prompts)
    print(json.dumps({"split_ready": True, "train": len(train), "validation": len(val), "prompts": len(prompts),
                      "text_device": args.device}), flush=True)
    text_cache(output, wan, device=args.device, batch_size=args.batch_size)
    files = {name: file_digest(output / name) for name in ["all.csv", "train.csv", "val.csv", "dataset_stats.json", "prompts.json", *prompts.values()]}
    summary = dict(passed=True, train=len(train), validation=len(val), frames=sum(e["length"] for e in episodes),
                   prompts=len(prompts), text_device=args.device, output=str(output))
    write_json(report_path, {"identity": provenance, "files": files, "summary": summary})
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vpp2-root", "source", "stats", "wan", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        raise ValueError("threads/batch-size must be positive")
    torch.set_num_threads(args.threads)
    bootstrap(args)


if __name__ == "__main__":
    main()
