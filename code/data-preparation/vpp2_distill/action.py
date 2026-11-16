"""Frozen Action2B with differentiable native cached attention."""
from __future__ import annotations

import torch
from torch import nn
import torch.utils.checkpoint

from .contracts import validate_native_config
from .native.action_dit import ActionDiT
from .native.checkpoint_compat import action_config
from .native.mot import MoT
from .native.scheduler import WanContinuousFlowMatchScheduler


class CachedActionRunner(MoT):
    """Reuse upstream MoT methods without constructing its required video expert."""
    def __init__(self, expert: ActionDiT, *, checkpoint=True):
        nn.Module.__init__(self)
        if expert.use_video_tokens:
            raise ValueError("The frozen expert must use native direct K/V")
        self.mixtures = nn.ModuleDict({"action": expert})
        self.expert_order = ["action"]
        self.num_layers = len(expert.blocks)
        self.num_heads = expert.num_heads
        self.attn_head_dim = expert.attn_head_dim
        self.use_video_tokens = False
        self.use_fixed_video_layers = None
        self.mot_checkpoint_mixed_attn = bool(checkpoint)
        expert.use_gradient_checkpointing = bool(checkpoint)
        expert.requires_grad_(False).eval()

    @property
    def expert(self):
        return self.mixtures["action"]

    def train(self, mode=True):
        super().train(mode)
        self.expert.eval()
        return self

    def forward(self, noisy_actions, timesteps, context, context_mask, video_kv):
        if len(video_kv) != self.num_layers:
            raise ValueError("Wrong number of video-memory layers")
        batch, horizon, _ = noisy_actions.shape
        video_len = video_kv[0]["k"].shape[1]
        shape = (batch, video_len, self.num_heads * self.attn_head_dim)
        for layer in video_kv:
            if set(layer) != {"k", "v"} or any(tuple(layer[k].shape) != shape for k in ("k", "v")):
                raise ValueError("Malformed direct video K/V")
        pre = self.expert.pre_dit(noisy_actions, timesteps, context, context_mask)
        # Same native joint mask: bidirectional video, all visible video tokens,
        # bidirectional action tokens, and no video-to-action attention.
        mask = torch.ones((video_len + horizon, video_len + horizon), dtype=torch.bool, device=noisy_actions.device)
        mask[:video_len, video_len:] = False
        tokens = self.forward_action_with_video_cache(
            action_tokens=pre["tokens"], action_freqs=pre["freqs"], action_t_mod=pre["t_mod"],
            action_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
            video_kv_cache=video_kv, attention_mask=mask, video_seq_len=video_len,
        )
        return self.expert.post_dit(tokens, pre)


class FrozenPolicyCore(nn.Module):
    def __init__(self, expert, proprio_encoder, *, checkpoint=True):
        super().__init__()
        self.runner = CachedActionRunner(expert, checkpoint=checkpoint)
        self.proprio_encoder = proprio_encoder.requires_grad_(False).eval()
        self.scheduler = WanContinuousFlowMatchScheduler(shift=3.0, num_train_timesteps=1000)

    @classmethod
    def from_checkpoint(cls, path, *, dtype=torch.bfloat16, checkpoint=True):
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        config = action_config(payload)
        validate_native_config(config)
        if payload.get("step") != 100000:
            raise ValueError("Expected paired joint step-100000 Action2B")
        kwargs = dict(config["model"]["action_dit_config"])
        kwargs["use_gradient_checkpointing"] = checkpoint
        expert = ActionDiT(**kwargs).to(dtype=dtype)
        expert.load_state_dict(payload["action_expert"], strict=True)
        proprio = nn.Linear(16, kwargs["text_dim"]).to(dtype=dtype)
        proprio.load_state_dict(payload["proprio_encoder"], strict=True)
        result = cls(expert, proprio, checkpoint=checkpoint)
        result.action_config = kwargs
        result.native_config = config
        return result

    def train(self, mode=True):
        super().train(mode)
        self.proprio_encoder.eval()
        self.runner.expert.eval()
        return self

    def append_proprio(self, context, mask, state):
        token = self.proprio_encoder(state.to(dtype=context.dtype).unsqueeze(1)).to(context.dtype)
        return (torch.cat((context, token), dim=1),
                torch.cat((mask, torch.ones((mask.shape[0], 1), dtype=torch.bool, device=mask.device)), dim=1))

    def forward(self, noisy_actions, timesteps, context, context_mask, video_kv):
        # Deliberately no no_grad(): frozen weights still differentiate inputs.
        return self.runner(noisy_actions, timesteps, context, context_mask, video_kv)

    def sample(self, initial_noise, context, mask, memory, *, trace=False):
        x = initial_noise
        times, deltas = self.scheduler.build_inference_schedule(10, x.device, x.dtype, shift_override=1.0)
        records = []
        for t, delta in zip(times, deltas):
            timestep = t.expand(x.shape[0])
            u = self(x, timestep, context, mask, memory)
            if trace:
                records.append({"x": x.detach().cpu(), "t": timestep.detach().cpu(), "u": u.detach().cpu()})
            x = self.scheduler.step(u, delta, x)
        return (x, records) if trace else x

