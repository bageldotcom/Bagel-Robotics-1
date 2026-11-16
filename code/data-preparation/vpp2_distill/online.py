"""Online-teacher client and deterministic inputs; never imports the Wan runtime."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch

from .contracts import (PREPROCESSING, UPSTREAM_COMMIT, cpu_noise, digest, file_digest,
                        seeded_int, tensor_digest)
from .dataset import digest_rollout, select_anchors
from .native.scheduler import WanContinuousFlowMatchScheduler

ONLINE_FORMAT = "vpp2-online-teacher-v1"
MAX_MESSAGE_BYTES = 32 << 20


def pack(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    result = stream.getvalue()
    if len(result) > MAX_MESSAGE_BYTES:
        raise ValueError("Online teacher message exceeds the size limit")
    return result


def unpack(payload):
    if not 0 < len(payload) <= MAX_MESSAGE_BYTES:
        raise ValueError("Invalid online teacher message size")
    value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError("Online teacher message must be a dictionary")
    return value


def validate_metadata(metadata, data):
    if metadata.get("format") != ONLINE_FORMAT or metadata.get("dataset_sha256") != data.manifest["sha256"]:
        raise ValueError("Online teacher dataset/protocol identity mismatch")
    identity = metadata["teacher_identity"]
    if (identity["stats_sha256"] != data.normalizer.identity or
            identity["preprocessing"] != PREPROCESSING or identity["upstream_commit"] != UPSTREAM_COMMIT):
        raise ValueError("Online teacher normalization/preprocessing/source identity mismatch")
    parity = metadata["native_parity"]
    if not parity["passed"] or parity["checked_steps"] != 10 or parity["teacher_identity"] != identity:
        raise ValueError("Online teacher requires matching native ten-step Action2B parity")
    return identity


def prepare_query(example, seed):
    """Construct ONE BF16 query, before either branch, without changing global RNG."""
    if not example.get("gt_valid", True):
        query = example["input_query"]
        return {"x": query["x"].detach().cpu().to(torch.bfloat16).clone(),
                "t": query["t"].detach().cpu().to(torch.bfloat16).reshape(()).clone(),
                "gt": torch.zeros((32, 16), dtype=torch.bfloat16)}
    scheduler = WanContinuousFlowMatchScheduler(shift=3.0, num_train_timesteps=1000)
    actions = example["action"].detach().cpu().to(torch.bfloat16).unsqueeze(0)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        t = scheduler.sample_training_t(1, torch.device("cpu"), torch.bfloat16)
    noise = cpu_noise(actions.shape, seeded_int(seed, "action-noise"))
    return {"x": scheduler.add_noise(actions, noise, t)[0], "t": t[0],
            "gt": scheduler.training_target(actions, noise, t)[0]}


def validate_request(request, metadata):
    if (request.get("format") != ONLINE_FORMAT or
            request.get("teacher_identity") != metadata["teacher_identity"] or
            request.get("dataset_sha256") != metadata["dataset_sha256"]):
        raise ValueError("Online request teacher/dataset identity mismatch")
    example, query = request["example"], request["query"]
    # Ground-truth labels never enter the teacher's video memory producer.
    if set(example) != {"history", "context", "context_mask", "state"} or set(query) != {"x", "t"}:
        raise ValueError("Unexpected fields in online conditioning/query")
    shapes = {"history": (9, 3, 240, 416), "context": (128, 4096), "context_mask": (128,), "state": (16,)}
    for key, shape in shapes.items():
        if not isinstance(example[key], torch.Tensor) or tuple(example[key].shape) != shape:
            raise ValueError(f"Invalid online {key} shape")
    if example["history"].dtype != torch.uint8 or example["context_mask"].dtype != torch.bool or not example["context_mask"].all():
        raise ValueError("Online history/text padding differs from native preprocessing")
    for key in ("context", "state"):
        if example[key].dtype != torch.bfloat16 or not torch.isfinite(example[key]).all():
            raise ValueError(f"Invalid online {key} dtype/value")
    for key, shape in (("x", (32, 16)), ("t", ())):
        value = query[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
            raise ValueError("Invalid online action query")
    if not 0 <= query["t"].item() <= 1000:
        raise ValueError("Action timestep is outside the native schedule")
    for key in ("video_seed", "query_seed"):
        if type(request[key]) is not int or not 0 <= request[key] < 2**63:
            raise ValueError("Invalid online random seed")


def validate_targets(target):
    kv, velocity = target["kv"], target["teacher"]
    if velocity.shape != (32, 16) or velocity.dtype != torch.bfloat16 or not torch.isfinite(velocity).all():
        raise ValueError("Invalid online teacher action response")
    if kv["layers"].shape != (4,) or kv["layers"].dtype != torch.int64 or (kv["layers"] // 10).tolist() != [0, 1, 2, 3]:
        raise ValueError("Online K/V targets must sample all four depth quartiles")
    if kv["tokens"].shape != (4, 32) or kv["tokens"].dtype != torch.int64 or not ((kv["tokens"] >= 0) & (kv["tokens"] < 1950)).all():
        raise ValueError("Invalid online K/V token indices")
    if any(len(row.unique()) != 32 for row in kv["tokens"]):
        raise ValueError("Online K/V token coordinates must be unique within each layer")
    for kind in ("k", "v"):
        if kv[kind].shape != (4, 32, 5120) or kv[kind].dtype != torch.bfloat16 or not torch.isfinite(kv[kind]).all():
            raise ValueError("Invalid online K/V target values")


class OnlineTeacherClient:
    def __init__(self, config, data):
        urls = config.get("urls", [])
        if not urls or not all(isinstance(url, str) and url.startswith(("http://", "https://")) for url in urls):
            raise ValueError("Set online_teacher.urls to the running teacher service endpoint(s)")
        self.timeout = float(config.get("timeout", 900))
        if self.timeout <= 0:
            raise ValueError("Online teacher timeout must be positive")
        self.url = urls[int(os.environ.get("RANK", "0")) % len(urls)].rstrip("/")
        # Check every configured replica, so rank routing cannot silently change teachers.
        metadata = [self._get_metadata(url.rstrip("/")) for url in urls]
        self.identity = validate_metadata(metadata[0], data)
        for item in metadata[1:]:
            if validate_metadata(item, data) != self.identity or item["key_norms_sha256"] != metadata[0]["key_norms_sha256"]:
                raise ValueError("Online teacher replicas use different checkpoints/key norms")
        self.metadata = metadata[0]

    def _request(self, path, payload=None, *, url=None):
        request = Request((url or self.url) + path, data=payload,
                          headers={"Content-Type": "application/octet-stream"} if payload is not None else {})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = response.read(MAX_MESSAGE_BYTES + 1)
                if len(result) > MAX_MESSAGE_BYTES:
                    raise ValueError("Online teacher response exceeds the size limit")
                return result
        except HTTPError as exc:
            detail = exc.read(4096).decode(errors="replace")
            raise RuntimeError(f"Online teacher rejected {path}: {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"Online teacher unavailable at {url or self.url}; no cached/GT-only fallback: {exc}") from exc

    def _get_metadata(self, url):
        return json.loads(self._request("/metadata", url=url))

    def key_norms(self):
        norms = unpack(self._request("/key-norms"))
        if any(norms.get(key) != value for key, value in self.identity.items()):
            raise ValueError("Online key norms belong to another teacher")
        weights = norms["weights"]
        if weights.shape != (40, 5120) or not torch.isfinite(weights).all() or tensor_digest(weights) != self.metadata["key_norms_sha256"]:
            raise ValueError("Online key-norm content mismatch")
        return norms

    def targets(self, example, query):
        inputs = {key: example[key].detach().cpu() for key in ("history", "context", "context_mask", "state")}
        inputs["context"] = inputs["context"].to(torch.bfloat16)
        inputs["state"] = inputs["state"].to(torch.bfloat16)
        request = dict(format=ONLINE_FORMAT, teacher_identity=self.identity,
                       dataset_sha256=self.metadata["dataset_sha256"], example=inputs,
                       video_seed=int(example["video_seed"]), query_seed=int(example["online_query_seed"]),
                       query={key: query[key] for key in ("x", "t")})
        validate_request(request, self.metadata)
        payload = pack(request)
        response = unpack(self._request("/targets", payload))
        if (response.get("format") != ONLINE_FORMAT or response.get("teacher_identity") != self.identity or
                response.get("request_sha256") != hashlib.sha256(payload).hexdigest()):
            raise ValueError("Online teacher response identity/query mismatch")
        validate_targets(response)
        return response

    def attach(self, examples):
        result = []
        for source in examples:
            example = dict(source)
            if "query" in example or "kv_target" in example:
                raise ValueError("Online training cannot mix precomputed teacher targets into a batch")
            if example.get("sample_valid", True):
                query = prepare_query(example, example["online_query_seed"])
                target = self.targets(example, query)
                example["query"] = {**query, "teacher": target["teacher"].detach()}
                example["kv_target"] = {key: value.detach() for key, value in target["kv"].items()}
            result.append(example)
        return result


class OnlineReferences:
    """Identity and a fixed validation panel; contains no saved teacher outputs."""
    def __init__(self, data, identity, *, seed=100008, validation_anchors=128):
        self.data, self.seed = data, seed
        self.indices = select_anchors(data, validation_anchors, seed) if data.split == "validation" else None
        manifest = dict(format=ONLINE_FORMAT, dataset_sha256=data.manifest["sha256"], teacher_identity=identity,
                        split=data.split, indices=self.indices, seed=seed, validation_video_seeds=2)
        self.manifest = {**manifest, "sha256": digest(manifest)}

    def __len__(self):
        return 2 * len(self.indices) if self.indices is not None else len(self.data)

    def get(self, index, query_index=0):
        # Validation uses fixed inputs across checkpoints, independent of train RNG.
        repeat, position = divmod(int(index), len(self.indices) if self.indices is not None else len(self.data))
        example = dict(self.data[self.indices[position] if self.indices is not None else position])
        example["video_seed"] = seeded_int(self.seed, "online-validation-video", position, repeat)
        example["online_query_seed"] = seeded_int(self.seed, "online-validation-action", position, repeat)
        return example


class OnlineRollouts:
    """Raw captured student histories/queries; teacher labels are computed online."""
    def __init__(self, root, data):
        self.root, self.data = Path(root), data
        self.records = []
        for path in sorted(self.root.glob("*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self._check(payload)
            self.records.append({"path": str(path), "sha256": file_digest(path), "kind": "rollout"})
        if not self.records:
            raise ValueError("Online refresh needs captured student rollout records")
        manifest = dict(format="vpp2-online-rollouts-v1", dataset_sha256=data.manifest["sha256"], records=self.records)
        self.manifest = {**manifest, "sha256": digest(manifest)}

    def _check(self, payload):
        if payload["format"] != "vpp2-student-rollout-v1" or payload["dataset_sha256"] != self.data.manifest["sha256"]:
            raise ValueError("Online rollout dataset identity mismatch")
        example = payload["example"]
        if example.get("gt_valid") is not False or payload["sample_sha256"] != digest_rollout(example):
            raise ValueError("Online rollouts must be verified teacher-only examples")
        if len(payload["queries"]) != 10 or any(set(q) != {"x", "t"} for q in payload["queries"]):
            raise ValueError("Online rollouts require the ten captured student action queries")

    def __len__(self):
        return len(self.records)

    def get(self, index, query_index=0):
        record = self.records[int(index)]
        if file_digest(record["path"]) != record["sha256"]:
            raise ValueError("Online rollout content checksum changed")
        payload = torch.load(record["path"], map_location="cpu", weights_only=True)
        self._check(payload)
        example = dict(payload["example"])
        example["input_query"] = payload["queries"][int(query_index) % len(payload["queries"])]
        return example
