"""Portable, content-addressed contracts shared by teacher and student environments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

UPSTREAM_COMMIT = "72c17acf022494ff63617be86079bb92e92f5bf8"
THAW_COMMIT = "e8589ccc501310f0061c6136b6fc103b627767b4"
PREPROCESSING = "native-tshape-bf16-roundtrip-anchor-history8-stride25-text-padding-v2"
DATA_FORMAT = "vpp2-distill-demonstrations-v1"
TARGET_FORMAT = "vpp2-distill-teacher-v1"
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def seeded_int(*parts) -> int:
    return int(digest(parts)[:15], 16)


def history_indices(frame: int, length: int) -> list[int]:
    if not 0 <= frame < length:
        raise ValueError("History frame is outside its episode")
    return [0] + [max(0, frame + offset) for offset in range(-175, 1, 25)]


def cpu_noise(shape, seed, *, device="cpu", dtype=torch.bfloat16):
    # Native serving draws FP32 on CPU before casting to the model dtype.
    return torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(int(seed)),
                       dtype=torch.float32).to(device=device, dtype=dtype)


def native_text_context(context, mask):
    """Match VPP2 encode_prompt and VideoEventDataset: zero pads, attend all slots."""
    mask = mask.to(device=context.device, dtype=torch.bool)
    if context.ndim != 2 or mask.shape != context.shape[:1] or not mask.any():
        raise ValueError("Invalid native text context/mask")
    return context.masked_fill(~mask[:, None], 0).contiguous(), torch.ones_like(mask)


@dataclass(frozen=True)
class MemoryConfig:
    qwen_dim: int = 2560
    context_dim: int = 4096
    width: int = 2048
    blocks: int = 8
    heads: int = 16
    ffn_dim: int = 8192
    video_layers: int = 40
    kv_heads: int = 40
    head_dim: int = 128
    grid: tuple[int, int, int] = (5, 15, 26)
    latent_channels: int = 16
    patch: int = 2
    eps: float = 1e-6

    @property
    def tokens(self):
        return int(np.prod(self.grid))

    @property
    def kv_dim(self):
        return self.kv_heads * self.head_dim

    @property
    def noise_shape(self):
        t, h, w = self.grid
        return (self.latent_channels, t, h * self.patch, w * self.patch)

    def to_dict(self):
        return asdict(self)


class EE16Normalizer:
    """Native z-score arithmetic, including its INPUT clamp (never output clipping)."""
    def __init__(self, stats):
        self.stats = json.loads(Path(stats).read_text()) if isinstance(stats, (str, Path)) else stats
        self.identity = digest(self.stats)
        self.fields = {}
        for field in ("action", "state"):
            values = self.stats[field]["default"]
            mean = torch.as_tensor(values["global_mean"], dtype=torch.float32)
            std = torch.as_tensor(values["global_std"], dtype=torch.float32)
            if mean.shape != (16,) or std.shape != (16,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
                raise ValueError("Expected finite native EE16 z-score statistics")
            self.fields[field] = (1.0 / (std + 1e-8), -mean / (std + 1e-8))

    def normalize(self, value, field):
        value = torch.as_tensor(value, dtype=torch.float32)
        scale, offset = (x.to(value.device) for x in self.fields[field])
        return (value * scale + offset).clamp(-5.0, 5.0)

    def denormalize_actions(self, value):
        value = torch.as_tensor(value).float()
        scale, offset = (x.to(value.device) for x in self.fields["action"])
        return (value - offset) / scale


def validate_native_config(config):
    model, data = config["model"], config["data"]["train"]
    action = model["action_dit_config"]
    required = {"hidden_dim": 1024, "action_dim": 16, "ffn_dim": 4096, "num_layers": 40,
                "num_heads": 40, "attn_head_dim": 128, "text_dim": 4096, "freq_dim": 256}
    if any(action.get(k) != v for k, v in required.items()):
        raise ValueError("Expected the released RoboDojo Action2B architecture")
    if action.get("use_video_tokens", False) or model.get("use_video_tokens", False) or model.get("use_fixed_video_layers") is not None:
        raise ValueError("Only native direct per-layer K/V conditioning is supported")
    for name in ("action_video_timestep_cutoff", "action_clean_first_frame_start_layer", "action_clean_first_frame_timestep_cutoff"):
        if model.get(name) is not None:
            raise ValueError(f"Unsupported native routing: {name}")
    if model.get("action_chunk_size", 1) != 1 or model.get("action_visible_video_frames", 5) != 5:
        raise ValueError("Expected all five video-latent frames and action_chunk_size=1")
    if (data["action_horizon"], data["condition_history_frames"], data["condition_history_stride"]) != (32, 8, 25):
        raise ValueError("Expected horizon32/history8/stride25")
    if list(data["video_size"]) != [240, 416] or data["video_resize_mode"] != "direct" or data.get("use_relative_action", False):
        raise ValueError("Expected native T-shape and absolute EE16 actions")
    video = model["video_dit_config"]
    if model.get("proprio_dim") != 16 or video.get("video_attention_mask_mode", "bidirectional") != "bidirectional":
        raise ValueError("Expected EE16 proprioception and bidirectional video attention")
    if (video.get("num_layers"), video.get("dim"), video.get("num_heads"), video.get("attn_head_dim"), list(video.get("patch_size", []))) != (40, 5120, 40, 128, [1,2,2]):
        raise ValueError("Expected the released Wan14B spatial/temporal interface")
    if not data.get("condition_include_episode_first", False) or not data.get("condition_history_target_prefix", False):
        raise ValueError("Expected native episode anchor and history prefix")
    if model["video_scheduler"].get("train_sampling_strategy") != "pure_noise" or model["action_scheduler"].get("train_shift") != 3.0:
        raise ValueError("Expected pure-noise video prefill and native action training shift")
    processor = data["processor"]
    if processor.get("norm_default_mode") != "z-score" or processor.get("norm_exception_mode") or processor.get("action_state_transforms") or processor.get("use_stepwise_action_norm", False):
        raise ValueError("Expected unmodified native EE16 normalization")

