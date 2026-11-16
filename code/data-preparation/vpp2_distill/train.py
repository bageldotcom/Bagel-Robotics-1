"""Staged full-demonstration distillation with native frozen Action2B."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, set_seed

from .batches import LogicalBatches, OnlineBatches
from .bundle import load_checkpoint_model, save_architecture
from .contracts import digest, write_json
from .checkpoints import CHECKPOINT_FORMAT, checkpoint_settings, prune_checkpoints, recipe_config, require_checkpoint_space, source_identity
from .dataset import Demonstrations, TeacherReferences
from .model import DistillationPolicy


def load_inputs(config, stage, refresh_roots=(), rollout_roots=()):
    data = Demonstrations(config["data_root"], "train")
    mode = config.get("teacher_mode", "cached")
    if mode not in ("cached", "online"):
        raise ValueError("teacher_mode must be cached or online")
    if mode == "online":
        from .online import OnlineTeacherClient, OnlineReferences, OnlineRollouts
        if refresh_roots:
            raise ValueError("Online refresh uses --refresh-rollouts, not precomputed --refresh-cache targets")
        settings = config.get("online_teacher", {})
        teacher = OnlineTeacherClient(settings, data)
        identity = teacher.identity
        references = OnlineReferences(data, identity, seed=config["seed"])
        val = Demonstrations(config["data_root"], "validation")
        val_refs = OnlineReferences(val, identity, seed=config["seed"],
                                    validation_anchors=int(settings.get("validation_anchors", 128)))
        refresh = [OnlineRollouts(path, data) for path in rollout_roots]
        cfg = {**config["model"], "identity": identity}
        cfg.pop("teacher_key_norms", None)
        batches = OnlineBatches(data, references, teacher, refresh_references=refresh,
                                seed=config["seed"], stage=stage)
        return data, references, val_refs, cfg, batches
    if rollout_roots:
        raise ValueError("Cached refresh requires --refresh-cache; --refresh-rollouts is for online mode")
    manifest = json.loads((Path(config["teacher_root"]) / "manifest.json").read_text())
    identity = manifest["teacher_identity"]
    if identity["stats_sha256"] != data.normalizer.identity:
        raise ValueError("Teacher and student normalization statistics differ")
    parity = json.loads((Path(config["teacher_root"]) / "native_parity.json").read_text())
    if not parity["passed"] or parity["teacher_identity"] != identity or parity["checked_steps"] != 10:
        raise ValueError("Native Action2B parity must pass before training")
    references = TeacherReferences(config["teacher_root"], data, identity)
    val = Demonstrations(config["data_root"], "validation")
    val_refs = TeacherReferences(config["teacher_root"], val, identity)
    refresh = [TeacherReferences(path, data, identity) for path in refresh_roots]
    if any(r.get("kind") != "rollout" for source in refresh for r in source.records):
        raise ValueError("Refresh caches must contain only teacher-labeled student rollouts")
    cfg = dict(config["model"])
    cfg["identity"] = identity
    cfg["teacher_key_norms"] = str(Path(config["teacher_root"]) / "key_norms.pt")
    batches = LogicalBatches(data, references, refresh_references=refresh, seed=config["seed"], stage=stage)
    return data, references, val_refs, cfg, batches


def schedule_multiplier(step, warmup, total):
    if step < warmup:
        return (step+1) / max(warmup, 1)
    fraction = min(1., (step-warmup) / max(total-warmup, 1))
    return .1 + .9 * .5 * (1 + math.cos(math.pi*fraction))


def optimizer_for(model, stage):
    groups = [{"params": [p for p in model.memory.parameters() if p.requires_grad], "lr":1e-4, "name":"memory"}]
    if stage != "warmup":
        groups.append({"params":[p for p in model.encoder.parameters() if p.requires_grad], "lr":5e-6, "name":"qwen"})
    optimizer = torch.optim.AdamW(groups, betas=(.9, .95), weight_decay=.01)
    frozen = {id(p) for p in model.action_core.parameters()}
    if any(id(p) in frozen for group in optimizer.param_groups for p in group["params"]):
        raise AssertionError("Frozen policy parameters entered the optimizer")
    return optimizer


@torch.no_grad()
def validate(model, references, accelerator, online_teacher=None):
    was_training = model.training
    model.eval()
    totals = torch.zeros(4, device=accelerator.device)
    steps = math.ceil(len(references) / accelerator.num_processes)
    for offset in range(steps):
        i = offset * accelerator.num_processes + accelerator.process_index
        e = references.get(i % len(references), i % 4)
        e["gt_scale"], e["teacher_scale"] = 1., 1.
        if online_teacher is not None:
            e = online_teacher.attach([e])[0]
        with accelerator.autocast():
            result = model([e])
        if i < len(references):
            totals[:3] += torch.stack([result[k].to(totals) for k in ("gt_loss", "teacher_action_loss", "kv_loss")])
            totals[3] += 1
    totals = accelerator.reduce(totals, reduction="sum")
    model.train(was_training)
    return {key: float(totals[i] / totals[3].clamp_min(1)) for i, key in enumerate(("gt_loss", "teacher_action_loss", "kv_loss"))}


def checkpoint(accelerator, model, output, progress, settings=None):
    settings = settings or checkpoint_settings({})
    published = Path(output) / f"step_{progress['step']:07d}"
    directory = Path(output) / f".step_{progress['step']:07d}.incomplete"
    exists = torch.tensor(int(published.exists() or directory.exists()) if accelerator.is_main_process else 0, device=accelerator.device)
    if accelerator.reduce(exists, reduction="sum").item():
        raise FileExistsError(f"Checkpoint already exists: {directory}")
    # Check on every rank so a disk failure does not strand peers at a barrier.
    require_checkpoint_space(output, settings)
    accelerator.save_state(str(directory), safe_serialization=False)
    accelerator.wait_for_everyone()
    weights = "pytorch_model.bin"
    if not (directory / weights).exists():
        # Sharded backends save their own resume state; also gather portable
        # weights so deployment never depends on the training backend.
        state = accelerator.get_state_dict(model)
        weights = "portable_model.pt"
        if accelerator.is_main_process:
            torch.save(state, directory / weights)
        del state
    if accelerator.is_main_process:
        save_architecture(accelerator.unwrap_model(model), directory)
        write_json(directory / "progress.json", {**progress, "weights":weights, "checkpoint_format":CHECKPOINT_FORMAT})
        directory.rename(published)
        removed = prune_checkpoints(Path(output), published, int(settings["keep_last"]))
        if removed:
            print(json.dumps({"pruned_checkpoints":removed}), flush=True)
    accelerator.wait_for_everyone()
    return published


def run(args):
    config = yaml.safe_load(Path(args.config).read_text())
    if getattr(args, "teacher_mode", None):
        config["teacher_mode"] = args.teacher_mode
    mode = config.get("teacher_mode", "cached")
    if getattr(args, "teacher_url", None):
        config.setdefault("online_teacher", {})["urls"] = args.teacher_url
    for name in ("data_root", "output") + (("teacher_root",) if mode == "cached" else ()):
        override = getattr(args, name, None)
        if override:
            config[name] = override
        if not config.get(name):
            raise ValueError(f"Set {name} in configuration or command line")
    if args.action_checkpoint:
        config["model"]["action_checkpoint"] = args.action_checkpoint
    data, references, val_refs, model_config, batches = load_inputs(
        config, args.stage, args.refresh_cache, getattr(args, "refresh_rollouts", []))
    online_teacher = getattr(batches, "online_teacher", None)
    total = 1000 if args.stage == "warmup" else batches.steps_per_epoch * (3 if args.stage == "main" else 1)
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("max_steps must be positive")
        total = min(total, args.max_steps)
    settings = checkpoint_settings(config)
    implementation = source_identity()
    signature = digest({"config":recipe_config(config), "implementation_sha256":implementation["sha256"],
                        "dataset":data.manifest["sha256"], "targets":references.manifest["sha256"],
                        "refresh":[r.manifest["sha256"] for r in batches.refresh_references]})
    if args.dry_run:
        print(json.dumps({"stage":args.stage, "updates":total, "all_demonstration_windows":len(data),
                          "teacher_records":len(references), "validation_records":len(val_refs),
                          "effective_batch":32, "reference_fraction":1. if online_teacher is not None or args.stage == "warmup" else .25,
                          "signature":signature, "teacher_mode":mode, "loads_teacher_in_trainer":False,
                          "online_teacher_urls":config.get("online_teacher", {}).get("urls", []) if mode == "online" else []}, indent=2))
        return
    microbatch = int(config.get("microbatch", 1))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if microbatch < 1 or 32 % (world*microbatch):
        raise ValueError("world size × microbatch must divide 32")
    accelerator = Accelerator(gradient_accumulation_steps=32//(world*microbatch),
                              mixed_precision="bf16",
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)])
    if accelerator.distributed_type == DistributedType.FSDP:
        raise ValueError("Use DDP or DeepSpeed ZeRO-2; FSDP wrapping is not supported by the native cached expert")
    plugin = accelerator.state.deepspeed_plugin
    if plugin is not None:
        ds = plugin.deepspeed_config
        if ds.get("zero_optimization", {}).get("stage") != 2:
            raise ValueError("The sharded training path requires DeepSpeed ZeRO-2")
        required = {"train_micro_batch_size_per_gpu":microbatch,
                    "gradient_accumulation_steps":32//(world*microbatch), "train_batch_size":32,
                    "gradient_clipping":1.0}
        for key, value in required.items():
            if ds.get(key, "auto") not in ("auto", value):
                raise ValueError(f"DeepSpeed {key} differs from the logical training batch")
            ds[key] = value
    set_seed(int(config["seed"]))
    previous = None
    same_stage = False
    if args.resume:
        model, previous = load_checkpoint_model(args.resume)
        if previous["teacher_identity"] != model_config["identity"] or previous["dataset_sha256"] != data.manifest["sha256"]:
            raise ValueError("Resume checkpoint belongs to another dataset or teacher")
        same_stage = previous["stage"] == args.stage and previous.get("refresh_round", 0) == (args.refresh_round if args.stage == "refresh" else 0)
        if same_stage and (previous["signature"] != signature or previous["world_size"] != world or previous["total_steps"] != total):
            raise ValueError("Exact resume requires the same configuration, duration and world size")
        if not same_stage:
            allowed = (args.stage == "main" and previous["stage"] == "warmup") or (
                args.stage == "refresh" and (
                    (args.refresh_round == 1 and previous["stage"] == "main") or
                    (args.refresh_round == 2 and previous["stage"] == "refresh" and previous.get("refresh_round") == 1)))
            if not allowed or previous["step"] != previous["total_steps"]:
                raise ValueError("A new stage requires completion of the immediately preceding stage")
    else:
        if args.stage != "warmup":
            raise ValueError("Main/refresh stages require --resume from the preceding stage")
        model = (DistillationPolicy.build(model_config, teacher_key_norms=online_teacher.key_norms())
                 if online_teacher is not None else DistillationPolicy.build(model_config))
    model.set_stage(args.stage)
    model.memory.checkpoint = True
    model.action_core.runner.mot_checkpoint_mixed_attn = True
    model.action_core.runner.expert.use_gradient_checkpointing = True
    if hasattr(model.encoder.model, "gradient_checkpointing_enable") and args.stage != "warmup":
        model.encoder.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    optimizer = optimizer_for(model, args.stage)
    warmup = 100 if args.stage == "warmup" else 500
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: schedule_multiplier(step, warmup, total))
    model, optimizer = accelerator.prepare(model, optimizer)
    accelerator.register_for_checkpointing(scheduler)
    start = 0
    if previous and same_stage:
        accelerator.load_state(args.resume)
        start = int(previous["step"])
    output = Path(config["output"])
    resolved = {**config, "model":model_config, "stage":args.stage, "refresh_round":args.refresh_round if args.stage == "refresh" else 0, "signature":signature, "implementation":implementation}
    existing = output / "run.json"
    if existing.exists():
        old = json.loads(existing.read_text())
        if any(old.get(key) != resolved.get(key) for key in ("signature", "stage", "refresh_round")):
            raise ValueError("Output directory belongs to another run; choose a new one")
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        write_json(existing, resolved)
    accelerator.wait_for_everyone()
    optimizer.zero_grad(set_to_none=True)
    stop_after = getattr(args, "stop_after_updates", None)
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after_updates must be positive")
    end = min(total, start + stop_after) if stop_after is not None else total
    stop_file = getattr(args, "stop_file", None)
    last_saved = None
    for step in range(start, end):
        # Stateless per-update seed plus deterministic data/query selection gives
        # exact same-stage resume without relying on DataLoader worker RNG state.
        set_seed(int(config["seed"]) + step*world + accelerator.process_index)
        started = time.perf_counter()
        metrics = torch.zeros(4, device=accelerator.device)
        microbatches = batches.local_microbatches(step, accelerator.process_index, world, microbatch)
        for examples in microbatches:
            with accelerator.accumulate(model):
                # Teacher requests complete before constructing the student graph.
                # attach() creates one shared x_t/t and fresh structured K/V targets.
                if online_teacher is not None:
                    examples = online_teacher.attach(examples)
                with accelerator.autocast():
                    result = model(examples)
                if not torch.isfinite(result["action_loss"]):
                    raise FloatingPointError("Non-finite distillation objective")
                accelerator.backward(result["action_loss"])
                metrics += torch.stack([result[k].detach().to(metrics) for k in ("action_loss", "gt_loss", "teacher_action_loss", "kv_loss")]) / len(microbatches)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
        metrics = accelerator.reduce(metrics, reduction="mean")
        log_update = step == start or (step+1) % 10 == 0
        peaks = None
        if log_update and torch.cuda.is_available():
            peaks = accelerator.gather(torch.tensor([torch.cuda.max_memory_allocated()], device=accelerator.device)).cpu().tolist()
        if accelerator.is_main_process and log_update:
            record = dict(step=step+1, stage=args.stage, seconds=time.perf_counter()-started,
                          **{k:float(metrics[i]) for i,k in enumerate(("loss","gt","teacher_action","kv"))})
            if peaks is not None:
                record["peak_cuda_bytes"] = max(peaks)
                record["peak_cuda_bytes_by_rank"] = peaks
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        epoch_end = args.stage != "warmup" and (step+1) % batches.steps_per_epoch == 0
        requested_stop = torch.tensor(int(bool(stop_file) and Path(stop_file).exists()), device=accelerator.device)
        requested_stop = bool(accelerator.reduce(requested_stop, reduction="sum").item())
        if (step+1) % int(settings["every_steps"]) == 0 or epoch_end or step+1 == end or requested_stop:
            validation = validate(model, val_refs, accelerator, online_teacher)
            progress = dict(step=step+1, total_steps=total, stage=args.stage, world_size=world,
                            refresh_round=args.refresh_round if args.stage == "refresh" else 0,
                            signature=signature, teacher_mode=mode, teacher_identity=model_config["identity"],
                            dataset_sha256=data.manifest["sha256"], validation=validation)
            last_saved = checkpoint(accelerator, model, output, progress, settings)
            if accelerator.is_main_process:
                print(json.dumps({"checkpoint":str(last_saved), "validation":validation}), flush=True)
        if requested_stop:
            break
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/robodojo_qwen4b_action2b.yaml")
    parser.add_argument("--stage", choices=("warmup","main","refresh"), required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--teacher-root", help="Offline target directory; required only in cached mode")
    parser.add_argument("--teacher-mode", choices=("cached", "online"))
    parser.add_argument("--teacher-url", action="append", help="Online worker URL; repeat for replicas")
    parser.add_argument("--action-checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--resume")
    parser.add_argument("--refresh-round", type=int, choices=(1,2), default=1)
    parser.add_argument("--refresh-cache", action="append", default=[])
    parser.add_argument("--refresh-rollouts", action="append", default=[], help="Raw student rollout directory for online refresh")
    parser.add_argument("--max-steps", type=int, help="Change total stage duration (smoke tests only)")
    parser.add_argument("--stop-after-updates", type=int, help="Save and stop after this many additional updates; preserve full LR schedule")
    parser.add_argument("--stop-file", help="When this path appears, checkpoint and exit after the current logical update")
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

