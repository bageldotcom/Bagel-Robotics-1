import copy
import unittest
import torch
from torch import nn
from vpp2.learned_width_kv.controller import MaskController, stable_topk, schedule
from vpp2.learned_width_kv.masking import install_masks, convert_norm, apply_gate
from vpp2.learned_width_kv.reliability import weights, action_losses
from vpp2.width_kv.slicing import standard_selection, slice_state
from vpp2.compact_kv.model import VideoBranch
from vpp2.compact_kv.blocks import install_blocks, TerminalCacheBlock
from vpp2.models.wan21_14b.wan_video_dit import WanVideoDiT


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(100008)
        torch.set_num_threads(1)

    def test_controller_schedule_budget_and_independent_layers(self):
        selection = standard_selection(width=6,ffn=10,teacher_width=12,teacher_ffn=20,layers=3)
        c=MaskController(selection,12,20,3,6,10)
        self.assertEqual(sum(p.numel() for p in c.parameters()),52)
        h,f=c.gates(0); self.assertTrue(torch.equal(h,torch.ones(12)))
        self.assertEqual(schedule(800)['temperature'],10.)
        self.assertEqual(schedule(200)['discrete_ramp'],0.)
        self.assertEqual(stable_topk(torch.zeros(8),4).tolist(),[0,1,2,3])
        loss,_=c.regularizers(800); loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in c.parameters()))
        before=c.selection()
        with torch.no_grad(): c.ffn_scores[0].neg_()
        after=c.selection()
        self.assertNotEqual(before['ffn_indices'][0],after['ffn_indices'][0])
        self.assertEqual(before['ffn_indices'][1:],after['ffn_indices'][1:])

    def test_weighted_norm_affine_shift_and_finite_difference(self):
        norm=nn.LayerNorm(6,eps=1e-6).double(); convert_norm(norm)
        with torch.no_grad(): norm.weight.normal_(); norm.bias.normal_()
        x=torch.randn(2,3,6,dtype=torch.float64)
        g=torch.tensor([1,0,1,0,1,0],dtype=torch.float64)
        kept=[0,2,4]
        target=nn.functional.layer_norm(x[...,kept],(3,),norm.weight[kept],norm.bias[kept],norm.eps)
        torch.testing.assert_close(norm(x,g)[...,kept],target,rtol=1e-9,atol=1e-10)
        z=apply_gate(norm(x,g)+.23,g)
        self.assertEqual(int(torch.count_nonzero(z[...,[1,3,5]])),0)
        soft=torch.full((6,),.63,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda gate: apply_gate(norm(x,gate),gate),
                                               (soft,),eps=1e-6,atol=1e-6,rtol=1e-5))
        with self.assertRaises(ValueError): norm(x,torch.zeros(6))

    def test_full_native_and_sliced_kv(self):
        args=dict(hidden_dim=12,ffn_dim=20,num_layers=3,num_heads=2,attn_head_dim=8,
                  freq_dim=8,text_dim=10,in_dim=8,out_dim=2,cross_attn_norm=True)
        native=WanVideoDiT(**args).float().eval()
        masked=install_masks(copy.deepcopy(native)).eval()
        install_blocks(native,()); native.blocks[-1].__class__=TerminalCacheBlock
        native=VideoBranch(native).eval()
        inp=dict(context=torch.randn(1,3,10),context_mask=torch.ones(1,3,dtype=torch.bool),
                 clip_fea=torch.randn(1,257,1280),condition_latents=torch.randn(1,2,1,4,4))
        noise=torch.randn(1,2,2,4,4); t=torch.tensor([1000.]); mask=torch.ones(8,8,dtype=torch.bool)
        with torch.no_grad():
            a=native(inp,noise,t,mask); b=masked(inp,noise,t,mask)
        for l,r in zip(a,b):
            for key in ('k','v'): torch.testing.assert_close(l[key],r[key],rtol=0,atol=0)
        selection=standard_selection(width=6,ffn=10,teacher_width=12,teacher_ffn=20,layers=3)
        c=MaskController(selection,12,20,3,6,10)
        compact=WanVideoDiT(**dict(args,hidden_dim=6,ffn_dim=10)).float().eval()
        state,_=slice_state(masked.video.state_dict(),{k:v.shape for k,v in compact.state_dict().items()},selection,12)
        compact.load_state_dict(state,strict=True)
        install_blocks(compact,()); compact.blocks[-1].__class__=TerminalCacheBlock
        compact=VideoBranch(compact).eval()
        with torch.no_grad():
            b=masked(inp,noise,t,mask,*c.gates(1000,'binary')); d=compact(inp,noise,t,mask)
        for l,r in zip(b,d):
            for key in ('k','v'):
                rel=(l[key]-r[key]).square().mean().sqrt()/r[key].square().mean().sqrt().clamp_min(1e-6)
                self.assertLess(float(rel),1e-5)
        masked.requires_grad_(False)
        outputs=masked(inp,noise,t,mask,*c.gates(500))
        sum(v.square().mean() for layer in outputs for v in layer.values()).backward()
        self.assertGreater(float(c.hidden_scores.grad.abs().sum()),0)
        self.assertTrue((c.ffn_scores.grad.abs().sum(-1)>0).all())
        self.assertFalse(any(p.grad is not None for p in masked.parameters()))

    def test_reliability_bad_teacher_and_degenerate_thresholds(self):
        table={'x':[dict(median=1.,q75=1.,q95=3.)]*4}
        t=torch.tensor([.5,2.,4.,1.,1.]); s=torch.tensor([.5,3.,8.,.1,2.]); z=torch.tensor([2.,3.,8.,2.,.1])
        w,_=weights(t,s,z,['x']*5,[0]*5,table)
        torch.testing.assert_close(w,torch.tensor([1.,.5,0.,0.,0.]))
        table={'x':[dict(median=0.,q75=0.,q95=0.)]*4}
        w,_=weights(torch.zeros(1),torch.zeros(1),torch.ones(1),['x'],[0],table)
        self.assertEqual(float(w),1.)
        with self.assertRaises(FloatingPointError): weights(torch.tensor([float('nan')]),torch.zeros(1),torch.ones(1),['x'],[0],table)
        student=torch.ones(2,3,2,requires_grad=True); teacher=torch.full_like(student,10.); target=torch.zeros_like(student)
        gt,kd,raw,w,_=action_losses(student,teacher,target,None,torch.ones(2),['x']*2,[0]*2,table)
        self.assertEqual(float(kd),0.); self.assertGreater(float(raw),0.)
        (gt+kd).backward(); self.assertGreater(float(student.grad.abs().sum()),0.)

    def test_accumulation_complete_loss_scaling(self):
        initial=torch.tensor([.2,.6]); examples=torch.arange(8.).reshape(8,1)
        grads=[]
        for micro in (1,2,4,8):
            parameter=nn.Parameter(initial.clone())
            for x in examples.split(micro):
                prediction=(x*parameter).square().mean()
                regularizer=10*(parameter.sum()-1).square()
                ((prediction+regularizer)/(8//micro)).backward()
            grads.append(parameter.grad)
        for g in grads[1:]: torch.testing.assert_close(g,grads[0],rtol=1e-6,atol=1e-6)


if __name__=='__main__': unittest.main()
