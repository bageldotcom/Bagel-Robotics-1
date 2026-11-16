"""Every demonstration window remains available, regardless of teacher-cache coverage."""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import (DATA_FORMAT, TARGET_FORMAT, PREPROCESSING, EE16Normalizer,
                        digest, file_digest, history_indices, native_text_context, seeded_int, tensor_digest)


class Demonstrations(Dataset):
    def __init__(self, root, split="train", cache_episodes=2):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["format"] != DATA_FORMAT or self.manifest["preprocessing"] != PREPROCESSING:
            raise ValueError("Incompatible demonstration preprocessing")
        expected = {k: v for k, v in self.manifest.items() if k != "sha256"}
        if digest(expected) != self.manifest["sha256"]:
            raise ValueError("Demonstration manifest identity mismatch")
        self.normalizer = EE16Normalizer(self.root / "dataset_stats.json")
        if self.normalizer.identity != self.manifest["stats_sha256"]:
            raise ValueError("Demonstration normalization changed")
        self.episodes = [e for e in self.manifest["episodes"] if e["split"] == split]
        if not self.episodes:
            raise ValueError(f"No episodes in split {split}")
        self.split = split
        self.ends = np.cumsum([e["length"] for e in self.episodes]).tolist()
        self.lookup = {str(e["id"]): i for i, e in enumerate(self.episodes)}
        self.cache = OrderedDict()
        self.cache_episodes = cache_episodes

    def __len__(self):
        return self.ends[-1]

    def resolve(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = bisect_right(self.ends, index)
        return episode, index - (self.ends[episode-1] if episode else 0)

    def index_of(self, episode_id, frame):
        i = self.lookup[str(episode_id)]
        if not 0 <= frame < self.episodes[i]["length"]:
            raise ValueError("Teacher record points outside an episode")
        return (self.ends[i-1] if i else 0) + frame

    def _episode(self, i):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        e = self.episodes[i]
        path = self.root / e["path"]
        arrays = {name: np.load(path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                  for name in ("frames", "actions", "states")}
        if arrays["frames"].shape != (e["length"], 3, 240, 416) or arrays["frames"].dtype != np.uint8:
            raise ValueError("Invalid native image array")
        for name in ("actions", "states"):
            if arrays[name].shape != (e["length"], 16):
                raise ValueError("Invalid EE16 array")
        if e.get("text_sha256") and file_digest(self.root / e["text"]) != e["text_sha256"]:
            raise ValueError("Prepared text context checksum changed")
        text = torch.load(self.root / e["text"], map_location="cpu", weights_only=True)
        if text["prompt"] != e["prompt"] or text["context"].ndim != 2 or text["context"].shape[-1] != 4096:
            raise ValueError("Text cache differs from episode prompt")
        text["context"], text["mask"] = native_text_context(text["context"], text["mask"])
        arrays["text"] = text
        self.cache[i] = arrays
        while len(self.cache) > self.cache_episodes:
            self.cache.popitem(last=False)
        return arrays

    def __getitem__(self, index):
        i, frame = self.resolve(int(index))
        e, arrays = self.episodes[i], self._episode(i)
        positions = np.minimum(np.arange(frame, frame+32), e["length"]-1)
        history = torch.from_numpy(np.array(arrays["frames"][history_indices(frame, e["length"])], copy=True))
        state = self.normalizer.normalize(np.array(arrays["states"][frame], copy=True), "state")
        action = self.normalizer.normalize(np.array(arrays["actions"][positions], copy=True), "action")
        result = dict(episode=str(e["id"]), frame=frame, task=e["task"], prompt=e["prompt"],
                      history=history, state=state, action=action,
                      action_is_pad=torch.arange(frame, frame+32) >= e["length"],
                      context=arrays["text"]["context"], context_mask=arrays["text"]["mask"].bool(),
                      gt_valid=True, video_seed=1000004 + frame // 24)
        return result

    def sample_identity(self, example):
        return digest({"dataset": self.manifest["sha256"], "episode": example["episode"], "frame": example["frame"],
                       "prompt": example["prompt"], "history": tensor_digest(example["history"]),
                       "state": tensor_digest(example["state"]), "action": tensor_digest(example["action"]),
                       "mask": tensor_digest(example["action_is_pad"]),
                       "text": tensor_digest(example["context"]), "text_mask": tensor_digest(example["context_mask"])})


class TeacherReferences(Dataset):
    def __init__(self, root, demonstrations, expected_identity=None):
        self.root = Path(root)
        self.demonstrations = demonstrations
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["format"] != TARGET_FORMAT or self.manifest["dataset_sha256"] != demonstrations.manifest["sha256"]:
            raise ValueError("Teacher dataset identity mismatch")
        if digest({k:v for k,v in self.manifest.items() if k != "sha256"}) != self.manifest["sha256"]:
            raise ValueError("Teacher manifest was modified")
        if expected_identity is not None and self.manifest["teacher_identity"] != expected_identity:
            raise ValueError("Teacher checkpoint/normalization/preprocessing identity mismatch")
        self.records = [r for r in self.manifest["records"] if r["split"] == demonstrations.split]
        if not self.records:
            raise ValueError("Teacher reference split is empty")

    def __len__(self):
        return len(self.records)

    def get(self, index, query_index=0):
        record = self.records[int(index)]
        path = self.root / record["path"]
        if file_digest(path) != record["sha256"]:
            raise ValueError("Teacher target content checksum changed")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["teacher_identity"] != self.manifest["teacher_identity"]:
            raise ValueError("Teacher record identity mismatch")
        if record.get("kind", "demonstration") == "rollout":
            example = payload["example"]
            if example.get("gt_valid") is not False or payload["sample_sha256"] != digest_rollout(example):
                raise ValueError("Invalid rollout teacher sample")
        else:
            index = self.demonstrations.index_of(record["episode"], record["frame"])
            example = self.demonstrations[index]
            if payload["sample_sha256"] != self.demonstrations.sample_identity(example):
                raise ValueError("Teacher record no longer matches its demonstration")
        if payload["sample_sha256"] != record["sample_sha256"]:
            raise ValueError("Teacher record does not match manifest")
        example = dict(example)
        q = payload["queries"]
        j = int(query_index) % q["x"].shape[0]
        example["query"] = {key: q[key][j] for key in ("x", "t", "gt", "teacher")}
        example["kv_target"] = payload["kv"]
        example["video_seed"] = int(payload["video_seed"])
        return example

    def __getitem__(self, index):
        return self.get(index)


def digest_rollout(example):
    return digest({key: tensor_digest(example[key]) for key in ("history", "state", "context", "context_mask")} |
                  {key: example[key] for key in ("prompt", "video_seed", "action_seed")})


def select_anchors(dataset, count=8192, seed=100008):
    """Round-robin tasks/episodes, stratified episode progress, no duplicate windows."""
    rng = np.random.default_rng(seed)
    groups = {}
    for i, episode in enumerate(dataset.episodes):
        groups.setdefault(episode["task"], []).append(i)
    for group in groups.values():
        rng.shuffle(group)
    counts = [0] * len(dataset.episodes)
    cursors = {task: 0 for task in groups}
    total = 0
    tasks = sorted(groups)
    target = min(count, len(dataset))
    if target <= 0:
        raise ValueError("Anchor count must be positive")
    while total < target:
        for task in tasks:
            group = groups[task]
            for _ in group:
                i = group[cursors[task] % len(group)]
                cursors[task] += 1
                if counts[i] < dataset.episodes[i]["length"]:
                    counts[i] += 1
                    total += 1
                    break
            if total == target:
                break
    selected = []
    for i, n in enumerate(counts):
        if n:
            edges = np.linspace(0, dataset.episodes[i]["length"], n+1).astype(int)
            for a, b in zip(edges[:-1], edges[1:]):
                frame = int(rng.integers(a, b))
                selected.append((dataset.ends[i-1] if i else 0) + frame)
    rng.shuffle(selected)
    return selected


def identity_collate(examples):
    return examples
