"""Conditioned structured memory in the frozen expert's exact K/V interface."""
from __future__ import annotations

import torch
from torch import nn
import torch.utils.checkpoint

from .contracts import MemoryConfig
from .native.wan_video_dit import WanRMSNorm, precompute_freqs_cis_3d, rope_apply


class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(cfg.width, eps=cfg.eps) for _ in range(3)])
        self.self_attn = nn.MultiheadAttention(cfg.width, cfg.heads, dropout=0, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(cfg.width, cfg.heads, dropout=0, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(cfg.width, cfg.ffn_dim), nn.GELU(), nn.Linear(cfg.ffn_dim, cfg.width))

    def forward(self, x, memory, valid):
        q = self.norms[0](x)
        x = x + self.self_attn(q, q, q, need_weights=False)[0]
        x = x + self.cross_attn(self.norms[1](x), memory, memory, key_padding_mask=~valid, need_weights=False)[0]
        return x + self.ffn(self.norms[2](x))


class StructuredMemory(nn.Module):
    def __init__(self, cfg: MemoryConfig, key_norm_weights=None, *, checkpoint=True):
        super().__init__()
        self.cfg = cfg
        self.checkpoint = checkpoint
        if cfg.width % cfg.heads or cfg.head_dim % 2 or cfg.head_dim < 6:
            raise ValueError("Invalid attention dimensions")
        self.qwen_projection = nn.Linear(cfg.qwen_dim, cfg.width)
        self.context_projection = nn.Linear(cfg.context_dim, cfg.width)
        self.noise_projection = nn.Linear(cfg.latent_channels * cfg.patch**2, cfg.width)
        self.type_embeddings = nn.Parameter(torch.empty(2, cfg.width))
        self.positions = nn.ParameterList([nn.Parameter(torch.empty(n, cfg.width)) for n in cfg.grid])
        nn.init.normal_(self.type_embeddings, std=0.02)
        for p in self.positions:
            nn.init.normal_(p, std=0.02)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.blocks)])
        self.output_norm = nn.LayerNorm(cfg.width, eps=cfg.eps)
        self.k_projections = nn.ModuleList([nn.Linear(cfg.width, cfg.kv_dim) for _ in range(cfg.video_layers)])
        self.v_projections = nn.ModuleList([nn.Linear(cfg.width, cfg.kv_dim) for _ in range(cfg.video_layers)])
        self.key_norms = nn.ModuleList([WanRMSNorm(cfg.kv_dim, cfg.eps) for _ in range(cfg.video_layers)])
        if key_norm_weights is not None:
            if tuple(key_norm_weights.shape) != (cfg.video_layers, cfg.kv_dim):
                raise ValueError("Key normalization weights do not match the native K/V interface")
            for norm, weight in zip(self.key_norms, key_norm_weights):
                norm.weight.data.copy_(weight)
        self.key_norms.requires_grad_(False)
        t, h, w = cfg.grid
        ft, fh, fw = precompute_freqs_cis_3d(cfg.head_dim, end=max(cfg.grid))
        frequencies = torch.cat((ft[:t, None, None].expand(t, h, w, -1),
                                 fh[None, :h, None].expand(t, h, w, -1),
                                 fw[None, None, :w].expand(t, h, w, -1)), dim=-1).reshape(t*h*w, 1, -1)
        # Do not register this complex tensor as a buffer: Module.to(dtype=bf16)
        # would discard its imaginary component. Native RoPE also stores plain tensors.
        self.frequencies = frequencies

    def forward(self, hidden, hidden_mask, context, context_mask, noise):
        cfg = self.cfg
        if tuple(noise.shape[1:]) != cfg.noise_shape:
            raise ValueError("Video noise does not match the teacher latent grid")
        memory = torch.cat((self.qwen_projection(hidden) + self.type_embeddings[0],
                            self.context_projection(context) + self.type_embeddings[1]), dim=1)
        valid = torch.cat((hidden_mask.bool(), context_mask.bool()), dim=1)
        if not valid.any(dim=1).all():
            raise ValueError("Each example needs valid conditioning tokens")
        b, c, t, _, _ = noise.shape
        _, h, w = cfg.grid
        p = cfg.patch
        patches = noise.reshape(b, c, t, h, p, w, p).permute(0, 2, 3, 5, 1, 4, 6).reshape(b, cfg.tokens, c*p*p)
        pos = (self.positions[0][:, None, None] + self.positions[1][None, :, None] + self.positions[2][None, None, :]).reshape(cfg.tokens, cfg.width)
        x = self.noise_projection(patches) + pos
        for block in self.blocks:
            if self.checkpoint and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(block, x, memory, valid, use_reentrant=False)
            else:
                x = block(x, memory, valid)
        x = self.output_norm(x)
        result = []
        for k, v, norm in zip(self.k_projections, self.v_projections, self.key_norms):
            raw_key = k(x)
            key = norm(raw_key).to(raw_key.dtype)
            result.append({"k": rope_apply(key, self.frequencies, cfg.kv_heads), "v": v(x)})
        return result

