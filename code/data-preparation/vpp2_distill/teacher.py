"""Native frozen teacher for offline export and the separate online worker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .action import CachedActionRunner
from .contracts import (TARGET_FORMAT, UPSTREAM_COMMIT, PREPROCESSING, cpu_noise, digest,
                        file_digest, seeded_int, write_json, validate_native_config)
from .dataset import Demonstrations, select_anchors, digest_rollout
from .prepare import require_upstream


def sample_kv(kv, query_seed):
    """Copy only selected coordinates to CPU; never retain the full video memory."""
    generator = torch.Generator(device="cpu").manual_seed(query_seed)
    layers = torch.tensor([10*i + int(torch.randint(10, (1,), generator=generator)) for i in range(4)])
    positions = torch.stack([torch.randperm(1950, generator=generator)[:32] for _ in layers])
    sampled = {"layers": layers, "tokens": positions}
    for kind in ("k", "v"):
        sampled[kind] = torch.stack([kv[int(layer)][kind][0, pos.to(kv[0][kind].device)].detach().cpu()
                                    for layer, pos in zip(layers, positions)])
    return sampled


class Teacher:
    def __init__(self, args, stats_sha256):
        require_upstream(args.vpp2_root)
        from vpp2.policy import RoboDojoVPP2Policy
        from .native.checkpoint_compat import action_config
        self.policy = RoboDojoVPP2Policy(args.action, args.wan, dataset_stats_path=args.stats,
                                       video_checkpoint_path=args.video, device=args.device)
        self.model = self.policy.model.eval().requires_grad_(False)
        payload = torch.load(args.action, map_location="cpu", mmap=True, weights_only=True)
        config = action_config(payload)
        validate_native_config(config)
        if payload["step"] != 100000:
            raise ValueError("Expected the paired joint step-100000 teacher")
        self.identity = dict(upstream_commit=UPSTREAM_COMMIT, action_sha256=file_digest(args.action),
                             video_sha256=file_digest(args.video), stats_sha256=stats_sha256,
                             text_encoder_sha256=file_digest(Path(args.wan) / "models_t5_umt5-xxl-enc-bf16.pth"),
                             preprocessing=PREPROCESSING, native_config_sha256=digest(config))
        self.parity_runner = CachedActionRunner(self.model.action_expert, checkpoint=False).eval()

    @torch.no_grad()
    def prefill(self, example, seed):
        model = self.model
        condition = example["history"].permute(1, 0, 2, 3).unsqueeze(0).float().div(255).mul(2).sub(1)
        condition = condition.to(device=model.device, dtype=model.torch_dtype)
        context = example["context"].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        mask = example["context_mask"].unsqueeze(0).to(model.device)
        context, mask = model._append_proprio_to_context(context, mask, example["state"].unsqueeze(0).to(model.device, model.torch_dtype))
        latents = model._encode_separate_condition_latents(condition, tiled=False)
        clip = model._encode_clip_fea_from_video(condition[:, :, :1])
        noise = model._sample_action_video_latents(first_frame_latents=latents, num_video_frames=17,
                    generator=torch.Generator(device="cpu").manual_seed(seed), rand_device="cpu")
        result = model._build_frozen_video_cache(latents=noise, timestep_video=torch.full((1,), 1000., device=model.device, dtype=model.torch_dtype),
                    context=context, context_mask=mask, condition_latents=latents, clip_fea=clip,
                    action_seq_len=32, fuse_vae_embedding_in_latents=False, return_video_prediction=False)
        kv, attention, sequence_length = result[:3]
        if sequence_length != 1950 or len(kv) != 40:
            raise ValueError("Unexpected native video memory interface")
        return kv, attention, context, mask

    @torch.no_grad()
    def velocity(self, x, t, cache):
        kv, attention, context, mask = cache
        return self.model._predict_action_noise_with_cache(x, t, context, mask, kv, attention, 1950)

    @torch.no_grad()
    def online_targets(self, example, *, video_seed, query_seed, query):
        cache = self.prefill(example, video_seed)
        x = query["x"].to(self.model.device, self.model.torch_dtype).reshape(1, 32, 16)
        t = query["t"].to(self.model.device, self.model.torch_dtype).reshape(1)
        prediction = self.velocity(x, t, cache)[0].detach().cpu()
        sampled = sample_kv(cache[0], query_seed)
        del cache  # Release full teacher K/V before the student's autograd graph.
        return {"kv": sampled, "teacher": prediction}

    @torch.no_grad()
    def check_parity(self, example, cache, seed):
        kv, _, context, mask = cache
        x = cpu_noise((1, 32, 16), seed, device=self.model.device, dtype=self.model.torch_dtype)
        times, deltas = self.model.infer_action_scheduler.build_inference_schedule(10, x.device, x.dtype, shift_override=1.)
        maximum = 0.
        for t, delta in zip(times, deltas):
            native = self.velocity(x, t[None], cache)
            actual = self.parity_runner(x, t[None], context, mask, kv)
            maximum = max(maximum, float((actual.float()-native.float()).abs().max()))
            if not torch.equal(actual, native):
                raise AssertionError(f"Native BF16 Action2B parity failed: max_abs={maximum}")
            x = self.model.infer_action_scheduler.step(native, delta, x)
        return dict(passed=True, checked_steps=10, max_abs_error=maximum, teacher_identity=self.identity,
                    torch_version=str(torch.__version__), dtype=str(x.dtype), device=str(x.device))

    @torch.no_grad()
    def audit_contexts(self, data):
        """Compare prepared contexts to native encoding; do not infer policy retention."""
        prompts, rows = set(), []
        for i, episode in enumerate(data.episodes):
            if episode["prompt"] in prompts:
                continue
            prompts.add(episode["prompt"])
            text = data._episode(i)["text"]
            actual, mask = self.model.encode_prompt([episode["prompt"]])
            expected = text["context"].unsqueeze(0).to(actual.device, actual.dtype)
            expected_mask = text["mask"].unsqueeze(0).to(mask.device)
            if actual.shape != expected.shape or not torch.equal(mask, expected_mask) or not torch.isfinite(actual).all():
                raise ValueError("Native text encoding shape/mask/finiteness differs from prepared contexts")
            delta = actual.float()-expected.float()
            rows.append(dict(prompt_sha256=digest(episode["prompt"]), bitwise_equal=torch.equal(actual,expected),
                             max_abs=float(delta.abs().max()),
                             relative_mse=float(delta.square().mean()/(actual.float().square().mean()+1e-6))))
        return dict(comparison_only=True, policy_retention_established=False, prompts=len(rows),
                    device=str(self.model.device), torch_version=str(torch.__version__),
                    all_bitwise_equal=all(r["bitwise_equal"] for r in rows),
                    max_abs=max(r["max_abs"] for r in rows), rows=rows)

    @torch.no_grad()
    def parity_fixture(self, example, cache, seed):
        """One startup diagnostic; never used as a training target cache."""
        kv, _, context, mask = cache
        x = cpu_noise((1, 32, 16), seed, device=self.model.device, dtype=self.model.torch_dtype)
        initial = x.detach().cpu()
        times, deltas = self.model.infer_action_scheduler.build_inference_schedule(10, x.device, x.dtype, shift_override=1.)
        predictions = []
        for t, delta in zip(times, deltas):
            prediction = self.velocity(x, t[None], cache)
            predictions.append(prediction.detach().cpu())
            x = self.model.infer_action_scheduler.step(prediction, delta, x)
        return dict(format="vpp2-runtime-action-parity-v1", teacher_identity=self.identity,
                    teacher_torch=str(torch.__version__), initial=initial, times=times.cpu(), deltas=deltas.cpu(),
                    velocities=predictions, final=x.detach().cpu(), context=context.detach().cpu(), mask=mask.detach().cpu(),
                    state=example["state"].reshape(1,16).to(dtype=self.model.torch_dtype).cpu(),
                    kv=[{kind:layer[kind].detach().cpu() for kind in ("k","v")} for layer in kv])

    @torch.no_grad()
    def record(self, example, *, video_seed, query_seed, input_queries=None, check_parity=False):
        cache = self.prefill(example, video_seed)
        kv = cache[0]
        sampled = sample_kv(kv, query_seed)
        actions = example["action"].unsqueeze(0).to(device=self.model.device, dtype=self.model.torch_dtype)
        scheduler = self.model.train_action_scheduler
        queries = {k: [] for k in ("x", "t", "gt", "teacher")}
        if input_queries is None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(query_seed)
                times = scheduler.sample_training_t(4, torch.device("cpu"), actions.dtype)
            input_queries = []
            for i, t in enumerate(times):
                noise = cpu_noise(actions.shape, query_seed+i+1, device=actions.device, dtype=actions.dtype)
                t = t.reshape(1).to(actions.device)
                input_queries.append({"x": scheduler.add_noise(actions, noise, t)[0], "t": t[0],
                                      "gt": scheduler.training_target(actions, noise, t)[0]})
        for query in input_queries:
            x = query["x"].to(actions.device, actions.dtype).reshape(1, 32, 16)
            t = query["t"].to(actions.device, actions.dtype).reshape(1)
            prediction = self.velocity(x, t, cache)
            for key, value in (("x", x[0]), ("t", t[0]), ("gt", query.get("gt", torch.zeros_like(x[0]))), ("teacher", prediction[0])):
                queries[key].append(value.detach().cpu())
        parity = self.check_parity(example, cache, query_seed) if check_parity else None
        return {"kv": sampled, "queries": {k: torch.stack(v) for k, v in queries.items()},
                "video_seed": int(video_seed), "teacher_identity": self.identity}, parity


def export(args):
    data = Demonstrations(args.data, "train")
    args.stats = str(Path(args.data) / "dataset_stats.json")
    teacher = Teacher(args, data.normalizer.identity)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    identity = {"teacher_identity": teacher.identity, "dataset_sha256": data.manifest["sha256"],
                "anchors": args.anchors, "seed": args.seed, "rollouts": None if args.rollouts is None else str(Path(args.rollouts).resolve())}
    if (root / "identity.json").exists() and json.loads((root/"identity.json").read_text()) != identity:
        raise ValueError("Cannot mix target export identities")
    write_json(root / "identity.json", identity)
    norms = torch.stack([b.self_attn.norm_k.weight.detach().cpu() for b in teacher.model.video_expert.blocks])
    torch.save({"weights": norms, **teacher.identity}, root / "key_norms.pt")
    parity_done = False
    records = []
    sources = []
    if args.rollouts:
        for path in sorted(Path(args.rollouts).glob("*.pt")):
            value = torch.load(path, map_location="cpu", weights_only=True)
            if value.get("format") != "vpp2-student-rollout-v1" or value["dataset_sha256"] != data.manifest["sha256"]:
                raise ValueError("Unknown or incompatible student rollout")
            example = value["example"]
            if example["gt_valid"] or value["sample_sha256"] != digest_rollout(example):
                raise ValueError("Rollout record is not a valid teacher-only sample")
            sources.append(("train", data, None, example, value["queries"]))
        if len(sources) > args.anchors:
            # Deterministic task-balanced bounded refresh selection.
            groups = {}
            for item in sources:
                groups.setdefault(item[3]["task"], []).append(item)
            sources = []
            while len(sources) < args.anchors and any(groups.values()):
                for task in sorted(groups):
                    if groups[task] and len(sources) < args.anchors:
                        sources.append(groups[task].pop(0))
    else:
        for split in ("train", "validation"):
            dataset = data if split == "train" else Demonstrations(args.data, split)
            indices = select_anchors(dataset, args.anchors, args.seed) if split == "train" else [
                dataset.index_of(e["id"], int(t)) for e in dataset.episodes
                for t in torch.linspace(0, e["length"]-1, min(16, e["length"])).long().unique().tolist()]
            sources.extend((split, dataset, index, None, None) for index in indices)
    if not sources:
        raise ValueError("No teacher records selected")
    for ordinal, (split, dataset, index, rollout_example, input_queries) in enumerate(sources):
        example = dataset[index] if rollout_example is None else rollout_example
        sample_hash = dataset.sample_identity(example) if rollout_example is None else digest_rollout(example)
        seeds = [example["video_seed"], seeded_int(args.seed, sample_hash, "video")] if rollout_example is None else [example["video_seed"]]
        for seed_idx, video_seed in enumerate(seeds):
            identifier = digest((sample_hash, video_seed, teacher.identity))
            path = root / "records" / f"{identifier}.pt"
            path.parent.mkdir(exist_ok=True)
            if path.exists() and parity_done:
                value = torch.load(path, map_location="cpu", weights_only=True)
                if value["sample_sha256"] != sample_hash or value["teacher_identity"] != teacher.identity:
                    raise ValueError("Existing target file has another identity")
            else:
                value, parity = teacher.record(example, video_seed=int(video_seed), query_seed=seeded_int(args.seed, identifier, "query"),
                                               input_queries=input_queries, check_parity=not parity_done)
                value["sample_sha256"] = sample_hash
                if rollout_example is not None:
                    value["example"] = example
                temporary = path.with_suffix(".tmp")
                torch.save(value, temporary)
                temporary.replace(path)
                if parity:
                    write_json(root / "native_parity.json", parity)
                    parity_done = True
            records.append(dict(path=path.relative_to(root).as_posix(), split=split,
                                kind="demonstration" if rollout_example is None else "rollout",
                                episode=example.get("episode", "rollout"), frame=example.get("frame", 0),
                                sample_sha256=sample_hash, video_seed=int(video_seed), sha256=file_digest(path)))
        print(f"Teacher targets {ordinal+1}/{len(sources)}", flush=True)
    manifest = dict(format=TARGET_FORMAT, dataset_sha256=data.manifest["sha256"], teacher_identity=teacher.identity, records=records)
    write_json(root / "manifest.json", {**manifest, "sha256": digest(manifest)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("vpp2-root", "data", "action", "video", "wan", "output"):
        parser.add_argument(f"--{key}", required=True)
    parser.add_argument("--anchors", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=100008)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rollouts", help="Student-collected development histories; creates teacher-only targets")
    export(parser.parse_args())


if __name__ == "__main__":
    main()
