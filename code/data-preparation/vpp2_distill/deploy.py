"""Native RoboDojo history and seeds around the student memory producer."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import numpy as np
import torch

from .bundle import load_bundle
from .contracts import DEFAULT_PROMPT, EE16Normalizer, digest, native_text_context, write_json
from .dataset import digest_rollout
from .native.history import NativeHistory


class TextContexts:
    def __init__(self, bundle, teacher_identity, service_url=None):
        self.root = Path(bundle)
        self.index = json.loads((self.root / "text_index.json").read_text())
        self.cache = {}
        self.identity = teacher_identity
        self.service_url = service_url

    def get(self, prompt):
        if prompt in self.cache:
            return self.cache[prompt]
        if prompt in self.index:
            value = torch.load(self.root / self.index[prompt], map_location="cpu", weights_only=True)
            if value["prompt"] != prompt:
                raise ValueError("Text cache prompt mismatch")
        else:
            if not self.service_url:
                raise ValueError("Uncached instruction requires the original UMT5 text service; set text_service_url")
            request = Request(self.service_url, data=json.dumps({"prompt":prompt}).encode(), headers={"Content-Type":"application/json"}, method="POST")
            with urlopen(request, timeout=120) as response:
                value = json.load(response)
            if value["encoder_sha256"] != self.identity["text_encoder_sha256"] or value["prompt"] != prompt:
                raise ValueError("Text service encoder/prompt differs from the teacher")
            raw = base64.b64decode(value["context_bf16"], validate=True)
            if len(raw) != 128*4096*2:
                raise ValueError("Unexpected UMT5 response size")
            value = {"context":torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(128,4096).clone(),
                     "mask":torch.tensor(value["mask"], dtype=torch.bool), "prompt":prompt}
        if tuple(value["context"].shape) != (128,4096) or tuple(value["mask"].shape) != (128,):
            raise ValueError("Expected native UMT5 context and mask")
        if not torch.isfinite(value["context"]).all() or not value["mask"].any():
            raise ValueError("Invalid text context")
        value["context"], value["mask"] = native_text_context(value["context"], value["mask"])
        self.cache[prompt] = value
        return value


class StudentPolicy(NativeHistory):
    def __init__(self, bundle, *, device="cuda", seed=1, text_service_url=None,
                 rollout_output=None, development_tag=None, model=None, manifest=None):
        if model is None:
            model, manifest = load_bundle(bundle, device=device)
        self.model, self.manifest = model, manifest
        self.normalizer = EE16Normalizer(Path(bundle) / "dataset_stats.json")
        if self.normalizer.identity != manifest["teacher_identity"]["stats_sha256"]:
            raise ValueError("Deployment normalization mismatch")
        self.texts = TextContexts(bundle, manifest["teacher_identity"], text_service_url)
        self.action_horizon, self.replan_steps = 32, 24
        self.action_dim = self.proprio_dim = self.output_action_dim = 16
        self.action_representation = "absolute_dual_arm_ee_pose_qwxyz_plus_gripper"
        self.num_inference_steps, self.sigma_shift = 10, 1.
        self.video_size, self.visual_layout = (240,416), "robodojo_tshape"
        self.per_camera_video_size = None
        self.history_condition_frames, self.history_stride = 8, 25
        self.history_include_episode_first, self.history_buffer_size = True, 177
        self.base_seed, self.video_seed_offset = int(seed), 1000003
        self._inference_index = 0
        self._request_context = {}
        for name in self._PER_ENV_STATE:
            setattr(self, name, {})
        self.rollout_output = Path(rollout_output) if rollout_output else None
        self.development_tag = development_tag
        self.latencies = []
        self._trace_counter = 0
        if self.rollout_output:
            if not development_tag:
                raise ValueError("Rollout collection requires an explicit development_tag, separate from final evaluation")
            self.rollout_output.mkdir(parents=True, exist_ok=True)

    def set_request_context(self, context):
        self._request_context = dict(context)

    @torch.no_grad()
    def _infer_action_chunk(self, observation, instruction):
        if self.model.device.type == "cuda":
            torch.cuda.synchronize(self.model.device)
        started = time.perf_counter()
        env = int(observation.get("env_idx", 0))
        image = self._latest_image_by_env.get(env)
        if image is None:
            self.observe(observation)
            image = self._latest_image_by_env[env]
        # Read the exact uint8 frames directly; converting the normalized BF16
        # history back to uint8 a second time would alter some pixel values.
        current = int(self._history_step_by_env[env])
        frames = dict(self._history_by_env[env])
        anchor = self._history_anchor_by_env[env]
        selected = [anchor] + [anchor if current+offset <= 0 else frames[current+offset]
                               for offset in range(-175, 1, 25)]
        action_seed, video_seed = self._seeds(env)
        prompt = DEFAULT_PROMPT.format(task=instruction)
        text = self.texts.get(prompt)
        raw_state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        if raw_state.shape != (16,):
            raise ValueError("Expected absolute EE16 robot state")
        example = dict(history=torch.stack(selected), prompt=prompt,
                       state=self.normalizer.normalize(raw_state, "state"),
                       context=text["context"], context_mask=text["mask"],
                       action_seed=action_seed, video_seed=video_seed,
                       task=observation.get("task_name", instruction), frame=current,
                       episode=f"{self.development_tag}:{env}", gt_valid=False,
                       action=torch.zeros(32,16), action_is_pad=torch.zeros(32,dtype=torch.bool))
        with torch.autocast(self.model.device.type, dtype=torch.bfloat16, enabled=self.model.device.type == "cuda"):
            result = self.model.predict_chunk([example], trace=bool(self.rollout_output))
        if self.model.device.type == "cuda":
            torch.cuda.synchronize(self.model.device)
        actions = result[0] if self.rollout_output else result
        denormalized = self.normalizer.denormalize_actions(actions[0].cpu()).numpy()
        self.latencies.append(time.perf_counter()-started)
        if self.rollout_output:
            self._save_trace(example, result[1])
        return denormalized

    def _save_trace(self, example, trace):
        # These are teacher-query inputs, never ground-truth action labels.
        self._trace_counter += 1
        key = digest((self.development_tag, self._request_context, self._trace_counter, digest_rollout(example)))
        target = self.rollout_output / f"{key}.pt"
        if target.exists():
            raise FileExistsError("Rollout record already exists")
        value = {"format":"vpp2-student-rollout-v1", "dataset_sha256":self.manifest["dataset_sha256"],
                 "development_tag":self.development_tag, "sample_sha256":digest_rollout(example),
                 "example":example, "queries":[{"x":r["x"][0], "t":r["t"][0]} for r in trace]}
        temporary = target.with_suffix(".tmp")
        torch.save(value, temporary)
        temporary.replace(target)

    def performance(self):
        return {"latency_scope":"history/text lookup or service, student memory, action sampling and denormalization; excludes transport, observe and trace writes",
                "replans":len(self.latencies), "latency_p50_seconds":float(np.median(self.latencies)) if self.latencies else None,
                "latency_p95_seconds":float(np.percentile(self.latencies,95)) if self.latencies else None,
                "parameters":sum(p.numel() for p in self.model.parameters()),
                "peak_cuda_bytes":torch.cuda.max_memory_allocated(self.model.device) if self.model.device.type == "cuda" else 0}


def get_model(args):
    return StudentPolicy(args.get("checkpoint_path") or args["ckpt_setting"], device=args.get("device", "cuda"),
                         seed=int(args.get("seed",1)), text_service_url=args.get("text_service_url"),
                         rollout_output=args.get("rollout_output"), development_tag=args.get("development_tag"))

