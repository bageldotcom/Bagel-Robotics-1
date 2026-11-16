"""Cache-producing blocks with real module forwards for ZeRO-3 hooks.

Full blocks retain native parameter names. K/V-only blocks physically omit Q/O,
cross-attention and FFN parameters. No action layer is removed or renumbered.
"""
import torch
from torch import nn

from vpp2.models.wan21_14b.wan_video_dit import DiTBlock, flash_attention, modulate, rope_apply
from vpp2.models.wan21_14b.mot import MoT
from .spec import validate_architecture


class CacheFullBlock(DiTBlock):
    compact_cache_block = True

    def forward(self, x, t_mod, freqs, context, context_mask, video_mask):
        shift, scale, gate, mlp_shift, mlp_scale, mlp_gate = MoT._split_modulation(self, t_mod)
        z = modulate(self.norm1(x), shift, scale)
        q = rope_apply(self.self_attn.norm_q(self.self_attn.q(z)), freqs, self.num_heads)
        k = rope_apply(self.self_attn.norm_k(self.self_attn.k(z)), freqs, self.num_heads)
        v = self.self_attn.v(z)
        mixed = flash_attention(q, k, v, self.num_heads, video_mask)
        x = self.gate(x, gate, self.self_attn.o(mixed))
        if context is not None:
            if context_mask is not None and context_mask.ndim == 3:
                context_mask = context_mask.unsqueeze(1)
            x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        x = self.gate(x, mlp_gate, self.ffn(modulate(self.norm2(x), mlp_shift, mlp_scale)))
        return x, k, v


class TerminalCacheBlock(CacheFullBlock):
    """Retain all parameters while emitting only terminal action-facing K/V.

    The updated final video state has no consumer in cache-only training. Its
    dead branch conflicts with non-reentrant checkpointing and ZeRO-3 backward
    parameter materialization. Earlier blocks still execute in full. The native
    state_dict and every action-facing cache remain unchanged.
    """

    def forward(self, x, t_mod, freqs, context=None, context_mask=None, video_mask=None):
        shift, scale, *_ = MoT._split_modulation(self, t_mod)
        z = modulate(self.norm1(x), shift, scale)
        k = rope_apply(self.self_attn.norm_k(self.self_attn.k(z)), freqs, self.num_heads)
        v = self.self_attn.v(z)
        return x, k, v


class KVOnlyBlock(nn.Module):
    compact_cache_block = True

    def __init__(self, original):
        super().__init__()
        self.num_heads = original.num_heads
        self.norm1 = original.norm1
        self.self_attn = nn.Module()
        for name in ("k", "v", "norm_k"):
            setattr(self.self_attn, name, getattr(original.self_attn, name))
        self.modulation = nn.Parameter(original.modulation[:, :2].detach().clone(), requires_grad=original.modulation.requires_grad)

    def forward(self, x, t_mod, freqs, context=None, context_mask=None, video_mask=None):
        modulation = self.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift, scale = (modulation + t_mod[..., :2, :]).chunk(2, dim=-2)
        if t_mod.ndim == 4:
            shift, scale = shift.squeeze(-2), scale.squeeze(-2)
        z = modulate(self.norm1(x), shift, scale)
        k = rope_apply(self.self_attn.norm_k(self.self_attn.k(z)), freqs, self.num_heads)
        v = self.self_attn.v(z)
        return x, k, v


def install_blocks(video, indices):
    indices = validate_architecture(indices, len(video.blocks))
    if any(isinstance(block, KVOnlyBlock) for block in video.blocks):
        raise ValueError("A video expert can only be structurally pruned once")
    for i, block in enumerate(video.blocks):
        if i in indices:
            video.blocks[i] = KVOnlyBlock(block)
        else:
            # Same native modules and state keys; forward now includes the K/V output.
            block.__class__ = CacheFullBlock
    video.compact_kv_indices = indices
    return video
