"""Mask-aware Wan computation with native parameter names and exportable surgery.

Gate tensors are explicit forward arguments, including checkpoint recomputation.
No mutable hooks or external mask Parameters are attached to the sharded model.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from bagel_robotics.models.backbone.wan21_14b.wan_video_dit import WanVideoDiT, DiTBlock, WanLayerNorm, sinusoidal_embedding_1d, modulate, rope_apply, flash_attention
from bagel_robotics.models.backbone.wan21_14b.mot import MoT


def apply_gate(x, gate):
    if gate is None:
        return x
    # FP32 multiplication preserves score gradients; activations remain BF16.
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    return (x.to(dtype) * gate.to(dtype)).to(x.dtype)


class MaskedLayerNorm(nn.LayerNorm):
    def forward(self, x, gate=None):
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        value = x.to(dtype)
        weight = None if self.weight is None else self.weight.to(dtype)
        bias = None if self.bias is None else self.bias.to(dtype)
        if gate is None:
            # WanLayerNorm reduces in FP32; CLIP's terminal nn.LayerNorm uses
            # its native kernel when all channels are open.
            if self.native_wan_norm:
                return F.layer_norm(value, self.normalized_shape, weight, bias, self.eps).to(x.dtype)
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        g = gate.to(dtype)
        mass = g.sum()
        valid_mass = torch.isfinite(mass) & (mass > 0)
        if mass.is_cuda:
            torch._assert_async(valid_mass, 'An empty/nonfinite hidden mask cannot be normalized')
        elif not valid_mass:
            raise ValueError('An empty/nonfinite hidden mask cannot be normalized')
        mean = (value * g).sum(-1, keepdim=True) / mass
        variance = ((value - mean).square() * g).sum(-1, keepdim=True) / mass
        value = (value - mean) * torch.rsqrt(variance + self.eps)
        if weight is not None:
            value = value * weight
        if bias is not None:
            value = value + bias
        return value.to(x.dtype)


def convert_norm(norm):
    norm.native_wan_norm = isinstance(norm, WanLayerNorm)
    norm.__class__ = MaskedLayerNorm


class MaskedEmbedding(nn.Sequential):
    def forward(self, x, gate=None):
        return apply_gate(self[2](apply_gate(self[1](self[0](x)), gate)), gate)


class MaskedTimeProjection(nn.Sequential):
    def forward(self, x, gate=None):
        value = self[1](self[0](x))
        if gate is not None:
            value = apply_gate(value.unflatten(-1, (6, gate.numel())), gate).flatten(-2)
        return value


class MaskedImageProjection(nn.Module):
    def forward(self, x, gate=None):
        p = self.proj
        x = p[3](p[2](p[1](p[0](x))))
        # Excluded coordinates cannot affect the retained coordinates' moments.
        return apply_gate(p[4](x, gate), gate)


class MaskedBlock(DiTBlock):
    compact_cache_block = True

    def forward(self, x, t_mod, freqs, context, context_mask, video_mask, hidden_gate=None, ffn_gate=None):
        shift, scale, gate, mlp_shift, mlp_scale, mlp_gate = MoT._split_modulation(self, t_mod)
        z = apply_gate(modulate(self.norm1(x, hidden_gate), shift, scale), hidden_gate)
        k = rope_apply(self.self_attn.norm_k(self.self_attn.k(z)), freqs, self.num_heads)
        v = self.self_attn.v(z)
        if self.terminal_cache_only:
            return x, k, v
        q = rope_apply(self.self_attn.norm_q(self.self_attn.q(z)), freqs, self.num_heads)
        mixed = flash_attention(q, k, v, self.num_heads, video_mask)
        x = self.gate(x, gate, apply_gate(self.self_attn.o(mixed), hidden_gate))
        if context is not None:
            if context_mask is not None and context_mask.ndim == 3:
                context_mask = context_mask.unsqueeze(1)
            z = apply_gate(self.norm3(x, hidden_gate), hidden_gate)
            x = x + apply_gate(self.cross_attn(z, context, ctx_mask=context_mask), hidden_gate)
        z = apply_gate(modulate(self.norm2(x, hidden_gate), mlp_shift, mlp_scale), hidden_gate)
        # One FFN gate, after GELU. Gating both linears would square soft gates.
        z = self.ffn[2](apply_gate(self.ffn[1](self.ffn[0](z)), ffn_gate))
        x = self.gate(x, mlp_gate, apply_gate(z, hidden_gate))
        return x, k, v


class MaskedVideo(WanVideoDiT):
    def pre_dit(self, x, timestep, context, context_mask=None, clip_fea=None,
                condition_latents=None, hidden_gate=None, **kwargs):
        if x.ndim != 5 or context.ndim != 3:
            raise ValueError('Native latent/context ranks changed')
        if x.shape[1] == self.out_dim:
            x = torch.cat([x, self._build_condition(x, condition_latents)], dim=1)
        if x.shape[1] != self.in_dim:
            raise ValueError('Native video conditioning channels changed')
        batch = x.shape[0]
        if timestep.shape != (batch,):
            raise ValueError('Require one native timestep per example')
        if context_mask is None:
            context_mask = torch.ones(context.shape[:2], dtype=torch.bool, device=context.device)
        if clip_fea is None:
            clip_fea = torch.zeros((batch,257,1280), dtype=context.dtype, device=context.device)
        if clip_fea.shape != (batch,257,1280):
            raise ValueError('Invalid native image tokens')
        x = self.patchify(x)
        f, h, w = x.shape[2:]
        x_tokens = apply_gate(x.flatten(2).transpose(1,2).contiguous(), hidden_gate)
        weight = self.time_embedding[0].weight
        time_input = sinusoidal_embedding_1d(self.freq_dim, timestep).to(device=weight.device, dtype=weight.dtype)
        t = self.time_embedding(time_input, hidden_gate)
        t_mod = self.time_projection(t, hidden_gate).unflatten(1,(6,self.hidden_dim))
        text = self.text_embedding(context, hidden_gate)
        image = self.img_emb(clip_fea.to(text), hidden_gate)
        image_mask = torch.ones((batch,257), dtype=torch.bool, device=context_mask.device)
        combined = torch.cat([image_mask, context_mask.bool()], dim=1)
        freqs = torch.cat([
            self.freqs[0][:f].view(f,1,1,-1).expand(f,h,w,-1),
            self.freqs[1][:h].view(1,h,1,-1).expand(f,h,w,-1),
            self.freqs[2][:w].view(1,1,w,-1).expand(f,h,w,-1)], dim=-1).reshape(f*h*w,1,-1).to(x.device)
        return dict(tokens=x_tokens, t=t, t_mod=t_mod, context=torch.cat([image,text],dim=1),
                    context_mask=combined.unsqueeze(1).expand(-1,f*h*w,-1), freqs=freqs,
                    meta=dict(grid_size=(f,h,w), tokens_per_frame=h*w, batch_size=batch))


class MaskAwareVideoBranch(nn.Module):
    def __init__(self, video):
        super().__init__()
        self.video = video

    def forward(self, inputs, noise, timestep, video_mask, hidden_gate=None, ffn_gates=None):
        pre = self.video.pre_dit(noise, timestep, inputs['context'], inputs['context_mask'],
                    inputs['clip_fea'], inputs['condition_latents'], hidden_gate=hidden_gate)
        x, caches = pre['tokens'], []
        for i, block in enumerate(self.video.blocks):
            ffn = None if ffn_gates is None or i == len(self.video.blocks)-1 else ffn_gates[i]
            args = (x,pre['t_mod'],pre['freqs'],pre['context'],pre['context_mask'],video_mask,hidden_gate,ffn)
            if self.training and self.video.use_gradient_checkpointing:
                x,k,v = checkpoint(block,*args,use_reentrant=False)
            else:
                x,k,v = block(*args)
            caches.append(dict(k=k,v=v))
        return caches


def install_masks(video):
    video.__class__ = MaskedVideo
    video.text_embedding.__class__ = MaskedEmbedding
    video.time_embedding.__class__ = MaskedEmbedding
    video.time_projection.__class__ = MaskedTimeProjection
    video.img_emb.__class__ = MaskedImageProjection
    convert_norm(video.img_emb.proj[4])
    for i, block in enumerate(video.blocks):
        block.__class__ = MaskedBlock
        block.terminal_cache_only = i == len(video.blocks)-1
        for norm in (block.norm1, block.norm2, block.norm3):
            convert_norm(norm)
    return MaskAwareVideoBranch(video)
