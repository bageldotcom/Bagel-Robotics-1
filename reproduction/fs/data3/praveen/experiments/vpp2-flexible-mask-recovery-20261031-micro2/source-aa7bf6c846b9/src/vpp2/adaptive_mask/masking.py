"""Native full video + all action caches, with physically equivalent hard masks."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from vpp2.learned_width_kv.masking import (install_masks as legacy_install,apply_gate,convert_norm,MaskedBlock,MaskedLayerNorm)
from vpp2.models.wan21_14b.wan_video_dit import (WanRMSNorm,I2VCrossAttention,Head,flash_attention,modulate,rope_apply)
from vpp2.models.wan21_14b.mot import MoT
from .execution import checkpoint_layer

class ExactOpenLayerNorm(MaskedLayerNorm):
    def forward(self,x,gate=None):
        result=super().forward(x,gate)
        if gate is None:return result
        native=super().forward(x,None)
        # Exact native all-one value; keep the mask-aware surrogate derivative.
        return torch.where((gate.detach()==1).all(),native.detach()+(result-result.detach()),result)

class MaskedRMSNorm(WanRMSNorm):
    def forward(self,x,gate=None):
        if gate is None:return super().forward(x)
        value=x.float();g=gate.float();mass=g.sum()
        torch._assert_async(torch.isfinite(mass)&(mass>0),'Empty/nonfinite head mask')
        value=value*torch.rsqrt((value.square()*g).sum(-1,keepdim=True)/mass+self.eps)
        result=value.to(x.dtype)*self.weight
        native=super().forward(x)
        return torch.where((gate.detach()==1).all(),native.detach()+(result-result.detach()),result)

class MaskedCrossAttention(I2VCrossAttention):
    def forward(self,x,ctx,ctx_mask=None,head_gate=None):
        if head_gate is None:return super().forward(x,ctx,ctx_mask)
        coords=head_gate.repeat_interleave(self.attn_head_dim)
        q=self.norm_q(self.q(x),coords)
        ki=self.norm_k_img(self.k_img(ctx[:,:257]),coords)
        k=self.norm_k(self.k(ctx[:,257:]),coords)
        # Complete heads are independent. Gate their summed image/text output
        # once, preserving a useful surrogate gradient for currently removed heads.
        image=flash_attention(q,ki,self.v_img(ctx[:,:257]),self.num_heads)
        text=flash_attention(q,k,self.v(ctx[:,257:]),self.num_heads,
                             ctx_mask=None if ctx_mask is None else ctx_mask[...,257:])
        return self.o(apply_gate(text+image,coords))

class Block(MaskedBlock):
    def forward(self,x,t_mod,freqs,context,context_mask,video_mask,hidden_gate=None,ffn_gate=None,cross_gate=None):
        shift,scale,gate,ms,mv,mg=MoT._split_modulation(self,t_mod)
        z=apply_gate(modulate(self.norm1(x,hidden_gate),shift,scale),hidden_gate)
        k=rope_apply(self.self_attn.norm_k(self.self_attn.k(z)),freqs,self.num_heads)
        v=self.self_attn.v(z)
        q=rope_apply(self.self_attn.norm_q(self.self_attn.q(z)),freqs,self.num_heads)
        x=self.gate(x,gate,apply_gate(self.self_attn.o(flash_attention(q,k,v,self.num_heads,video_mask)),hidden_gate))
        if context is not None:
            if context_mask is not None and context_mask.ndim==3:context_mask=context_mask.unsqueeze(1)
            z=apply_gate(self.norm3(x,hidden_gate),hidden_gate)
            x=x+apply_gate(self.cross_attn(z,context,ctx_mask=context_mask,head_gate=cross_gate),hidden_gate)
        z=apply_gate(modulate(self.norm2(x,hidden_gate),ms,mv),hidden_gate)
        z=self.ffn[2](apply_gate(self.ffn[1](self.ffn[0](z)),ffn_gate))
        x=self.gate(x,mg,apply_gate(z,hidden_gate))
        return x,k,v

class MaskedHead(Head):
    def forward(self,x,t,hidden_gate=None):
        if hidden_gate is None:return super().forward(x,t)
        if t.ndim==3:
            shift,scale=(self.modulation.unsqueeze(0).to(t)+t.unsqueeze(2)).chunk(2,dim=2)
            shift,scale=shift.squeeze(2),scale.squeeze(2)
        else:shift,scale=(self.modulation.to(t)+t.unsqueeze(1)).chunk(2,dim=1)
        return self.head(apply_gate(self.norm(x,hidden_gate)*(1+scale)+shift,hidden_gate))

class VideoBranch(nn.Module):
    def __init__(self,video):super().__init__();self.video=video
    def forward(self,inputs,noise,timestep,video_mask,hidden_gate=None,ffn_gates=None,cross_gates=None,*,return_video=True):
        pre=self.video.pre_dit(noise,timestep,inputs['context'],inputs['context_mask'],inputs['clip_fea'],inputs['condition_latents'],hidden_gate=hidden_gate)
        x=pre['tokens'];caches=[]
        for i,block in enumerate(self.video.blocks):
            f=None if ffn_gates is None or i==len(self.video.blocks)-1 else ffn_gates[i]
            c=None if cross_gates is None or i==len(self.video.blocks)-1 else cross_gates[i]
            args=(x,pre['t_mod'],pre['freqs'],pre['context'],pre['context_mask'],video_mask,hidden_gate,f,c)
            if self.training and torch.is_grad_enabled() and checkpoint_layer(self.video,i):
                x,k,v=checkpoint(block,*args,use_reentrant=False)
            else:x,k,v=block(*args)
            caches.append(dict(k=k,v=v))
        if not return_video:return caches
        video=self.video.unpatchify(self.video.head(x,pre['t'],hidden_gate),pre['meta']['grid_size'])
        return dict(cache=caches,video=video)


def install(video):
    if not all(isinstance(b.cross_attn,I2VCrossAttention) for b in video.blocks):raise ValueError('Requires Wan I2V image/text cross attention')
    legacy_install(video)
    for b in video.blocks:
        b.__class__=Block;b.terminal_cache_only=False
        for norm in (b.norm1,b.norm2,b.norm3):norm.__class__=ExactOpenLayerNorm
        b.cross_attn.__class__=MaskedCrossAttention
        for n in (b.cross_attn.norm_q,b.cross_attn.norm_k,b.cross_attn.norm_k_img):
            if not isinstance(n,WanRMSNorm):raise ValueError('Require native Q/K normalization')
            n.__class__=MaskedRMSNorm
    video.head.__class__=MaskedHead;convert_norm(video.head.norm)
    video.head.norm.__class__=ExactOpenLayerNorm;video.img_emb.proj[4].__class__=ExactOpenLayerNorm
    return VideoBranch(video)
