"""Native input preparation and differentiable student-video-only execution."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from hydra.utils import instantiate
from omegaconf import OmegaConf

from bagel_robotics.models.checkpoint_compat import action_config
from bagel_robotics.models.backbone.wan21_14b.wan_video_dit import WanVideoDiT, precompute_freqs_cis_3d
from bagel_robotics._internal.compact_kv.spec import stable_seed


def native_config(config):
    from importlib.resources import files
    cfg = OmegaConf.load(str(files("bagel_robotics.models").joinpath("configs/native_training.yaml")))
    root = Path(config["root"])
    cfg.paths.data = str(root / "data/prepared")
    cfg.paths.media = str(root / "data/media")
    cfg.paths.wan = config["wan_root"]
    cfg.paths.video_init = str(Path(config["teacher_bundle"]) / "video.pt")
    cfg.paths.action_init = None
    return cfg


def load_reference(config, device="cuda", text=False):
    bundle = Path(config["teacher_bundle"])
    payload = torch.load(bundle / "action.pt", map_location="cpu", weights_only=True, mmap=True)
    cfg = OmegaConf.create(action_config(payload)).model
    cfg.model_id = config["wan_root"]
    cfg.tokenizer_model_id = str(Path(config["wan_root"]) / "google/umt5-xxl")
    cfg.dit_checkpoint_path = str(bundle / "video.pt")
    cfg.action_dit_pretrained_path = None
    cfg.load_text_encoder = text
    cfg.video_dit_config.use_gradient_checkpointing = False
    cfg.action_dit_config.use_gradient_checkpointing = True
    model = instantiate(cfg, device=device, model_dtype=torch.bfloat16)
    model.action_expert.load_state_dict(payload["action_expert"], strict=True)
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    model.requires_grad_(False).eval()
    if model.train_video_scheduler.train_sampling_strategy != "pure_noise":
        raise ValueError("Model requires native pure-noise video conditioning")
    return model






@torch.no_grad()
def prepare_batch(reference, sample, seed, *, encode_video_target=False, return_noise_u=False, prepared_inputs=None):
    # Keep native conditioning, masks and normalization, but avoid computing a
    # clean target that the action/KV-only objective never reads. The legacy
    # branch remains available for exact-equivalence and timing checks.
    if prepared_inputs is None:
        inputs = reference.build_inputs(sample, training_stage="joint", encode_video_target=encode_video_target)
    else:
        if encode_video_target or prepared_inputs.get('input_latents') is not None:
            raise ValueError('Cached native inputs are only valid for action/KV-only preparation')
        inputs = dict(prepared_inputs)
        if inputs['condition_latents'].shape[2] != 3 or inputs['action'].shape[1] != 32:
            raise ValueError('Invalid cached native geometry')
    if encode_video_target:
        shape = inputs["input_latents"].shape
    else:
        condition = inputs["condition_latents"]
        factor = int(reference.vae.temporal_downsample_factor)
        frames = int(inputs["num_video_frames"])
        if factor <= 0 or (frames - 1) % factor:
            raise ValueError("Native video geometry is incompatible with the VAE temporal factor")
        shape = (condition.shape[0], condition.shape[1], (frames - 1) // factor + 1,
                 condition.shape[3], condition.shape[4])
    if tuple(shape[1:]) != (16, 5, 30, 52):
        raise ValueError(f"Expected five native latent planes, got {shape}")
    positions = sample["pilot_position"].tolist()
    video_noise, action_noise, timesteps, noise_u = [], [], [], []
    for position in positions:
        gen = torch.Generator(device=inputs["action"].device)
        gen.manual_seed(stable_seed(seed, "noise", int(position)))
        video_noise.append(torch.randn(shape[1:], device=inputs["action"].device, dtype=inputs["action"].dtype, generator=gen))
        action_noise.append(torch.randn(inputs["action"].shape[1:], device=inputs["action"].device, dtype=inputs["action"].dtype, generator=gen))
        u = torch.rand((), device=inputs["action"].device, generator=gen)
        noise_u.append(u)
        scheduler = reference.train_action_scheduler
        timesteps.append((scheduler._phi(u, scheduler.shift) * scheduler.num_train_timesteps).to(inputs["action"].dtype))
    noise_v, noise_a, t_a = torch.stack(video_noise), torch.stack(action_noise), torch.stack(timesteps)
    t_v = torch.full((len(positions),), 1000, device=noise_v.device, dtype=noise_v.dtype)
    noisy_action = reference.train_action_scheduler.add_noise(inputs["action"], noise_a, t_a)
    target = reference.train_action_scheduler.training_target(inputs["action"], noise_a, t_a)
    # Clean future video targets are intentionally unused by this action/KV pilot.
    del inputs["input_latents"]
    joint_mask = reference._build_mot_attention_mask(1950, inputs["action"].shape[1], 390, noise_v.device)
    prepared = (inputs, noise_v, t_v, noisy_action, t_a, target, joint_mask)
    return (*prepared, torch.stack(noise_u)) if return_noise_u else prepared




def action_prediction(reference, inputs, cache, noisy_action, timestep, mask):
    pre = reference.action_expert.pre_dit(action_tokens=noisy_action, timestep=timestep,
            context=inputs["context"], context_mask=inputs["context_mask"])
    out = reference.mot.forward_action_with_video_cache(
        action_tokens=pre["tokens"], action_freqs=pre["freqs"], action_t_mod=pre["t_mod"],
        action_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
        video_kv_cache=cache, attention_mask=mask, video_seq_len=1950)
    return reference.action_expert.post_dit(out, pre)
