"""Deterministic logical batches; auxiliary losses stay correctly scaled across ranks."""
from __future__ import annotations
import math
import numpy as np

from .contracts import seeded_int


class LogicalBatches:
    def __init__(self, demonstrations, references, *, refresh_references=(), seed=100008,
                 stage="main", batch_size=32, reference_count=8):
        if batch_size != 32 or reference_count != 8:
            raise ValueError("Initial experiment fixes logical batch32 / teacher8")
        self.data, self.references = demonstrations, references
        self.refresh_references = list(refresh_references)
        self.seed, self.stage = seed, stage
        self.steps_per_epoch = math.ceil(len(demonstrations) / 24)
        if stage == "refresh" and not self.refresh_references:
            raise ValueError("Refresh training needs student-rollout teacher labels")
        self._epoch, self._order = None, None

    def _reference(self, source, rng, *, gt_valid=True):
        return dict(source=source, index=int(rng.integers(len(source))), query=int(rng.integers(2**31)),
                    gt_valid=gt_valid, sample_valid=True)

    def plan(self, step):
        """Choose the global batch without reading another rank's images or targets."""
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step, {"warmup":0,"main":1,"refresh":2}[self.stage]]))
        if self.stage == "warmup":
            examples = [self._reference(self.references, rng) for _ in range(32)]
        else:
            epoch, offset = divmod(step, self.steps_per_epoch)
            if self._epoch != epoch:
                self._order = np.random.default_rng(np.random.SeedSequence([self.seed, epoch])).permutation(len(self.data))
                self._epoch = epoch
            selected = self._order[offset*24:(offset+1)*24]
            examples = [dict(source=self.data, index=int(i), gt_valid=True, sample_valid=True) for i in selected]
            for e in examples:
                e["video_seed"] = int(rng.integers(1, 2**62))
            while len(examples) < 24:
                examples.append(dict(source=self.data, index=int(selected[0]), gt_valid=True, sample_valid=False))
            for i in range(8):
                source = self.references
                if self.stage == "refresh" and i >= 4:
                    source = self.refresh_references[int(rng.integers(len(self.refresh_references)))]
                examples.append(self._reference(source, rng, gt_valid=not (self.stage == "refresh" and i >= 4)))
        n_gt = sum(e.get("gt_valid", True) and e.get("sample_valid", True) for e in examples)
        n_teacher = sum("query" in e for e in examples)
        rng.shuffle(examples)
        for e in examples:
            e["gt_scale"] = 32 / max(n_gt, 1)
            e["teacher_scale"] = 32 / max(n_teacher, 1)
        return examples

    def example(self, spec):
        source, index = spec["source"], spec["index"]
        example = source.get(index, spec["query"]) if "query" in spec else source[index]
        if example.get("gt_valid", True) != spec["gt_valid"]:
            raise ValueError("Teacher-only and ground-truth sources were mixed")
        for key in ("sample_valid", "video_seed", "gt_scale", "teacher_scale"):
            if key in spec:
                example[key] = spec[key]
        return example

    def get(self, step, rank=0, world_size=1):
        if not 0 <= rank < world_size or 32 % world_size:
            raise ValueError("Invalid rank/world size for logical batch32")
        return [self.example(spec) for spec in self.plan(step)[rank::world_size]]

    def local_microbatches(self, step, rank=0, world_size=1, microbatch=1):
        if 32 % (world_size * microbatch):
            raise ValueError("world_size * microbatch must divide effective batch size 32")
        examples = self.get(step, rank, world_size)
        return [examples[i:i+microbatch] for i in range(0, len(examples), microbatch)]



class OnlineBatches(LogicalBatches):
    """Full-stream supervision: 32 demos, or 28 demos + 4 raw rollout queries."""
    def __init__(self, demonstrations, references, teacher, *, refresh_references=(), seed=100008, stage="main"):
        super().__init__(demonstrations, references, refresh_references=refresh_references, seed=seed, stage=stage)
        self.online_teacher = teacher
        self.demonstration_count = 28 if stage == "refresh" else 32
        self.steps_per_epoch = math.ceil(len(demonstrations) / self.demonstration_count)

    def plan(self, step):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step, {"warmup":0, "main":1, "refresh":2}[self.stage]]))
        epoch, offset = divmod(step, self.steps_per_epoch)
        if self._epoch != epoch:
            self._order = np.random.default_rng(np.random.SeedSequence([self.seed, epoch])).permutation(len(self.data))
            self._epoch = epoch
        count = self.demonstration_count
        selected = self._order[offset*count:(offset+1)*count]
        examples = [dict(source=self.data, index=int(i), gt_valid=True, sample_valid=True) for i in selected]
        while len(examples) < count:
            examples.append(dict(source=self.data, index=int(selected[0]), gt_valid=True, sample_valid=False))
        if self.stage == "refresh":
            for _ in range(4):
                source = self.refresh_references[int(rng.integers(len(self.refresh_references)))]
                examples.append(self._reference(source, rng, gt_valid=False))
        n_gt = sum(e["gt_valid"] and e["sample_valid"] for e in examples)
        n_teacher = sum(e["sample_valid"] for e in examples)
        for position, example in enumerate(examples):
            example["online_query_seed"] = seeded_int(self.seed, self.stage, step, position, "query")
            # Captured student rollouts must retain their ORIGINAL video seed.
            if example["gt_valid"]:
                example["video_seed"] = seeded_int(self.seed, self.stage, step, position, "video")
            example["gt_scale"] = 32 / max(n_gt, 1)
            example["teacher_scale"] = 32 / max(n_teacher, 1)
        rng.shuffle(examples)
        return examples

    def example(self, spec):
        example = super().example(spec)
        example["online_query_seed"] = spec["online_query_seed"]
        return example
