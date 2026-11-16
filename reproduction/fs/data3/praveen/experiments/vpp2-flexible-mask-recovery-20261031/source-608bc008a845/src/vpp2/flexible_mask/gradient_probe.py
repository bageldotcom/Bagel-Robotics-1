"""Measure mask derivatives without touching video optimizer accumulation.

DeepSpeed engine outputs carry backward hooks that can scale gradients and flush
partitioned buffers. Diagnostic passes call the sharded MODULE and request
derivatives only for the masks. Keep the video requires_grad flags unchanged:
ZeRO checkpoint/recomputation hooks rely on those flags to retain fetched
parameters. autograd.grad with mask-only inputs does not accumulate parameter
gradients or invoke their post-accumulation hooks. Exact distributed
noninterference is checked before training is permitted.
"""
import torch
from .rng import diagnostic_rng


def frozen_weight_mask_gradient(module,masks,objective,args):
    parameters=list(module.parameters());flags=[p.requires_grad for p in parameters]
    parameter_gradients=[(p.grad,None if p.grad is None else p.grad._version) for p in parameters]
    before=[None if p.grad is None else p.grad.detach().clone() for p in masks]
    # Outside activation-checkpointed blocks (e.g. conditioning projections),
    # mask-only autograd can outlive the ZeRO parameter fetch. Preserve saved
    # parameter VALUES, not the mutable Parameter whose data becomes an empty
    # shard. Checkpointed blocks use their own saved-tensor hooks, so this does
    # not duplicate the whole video model. Repeated saves share one value copy.
    saved_parameters={}
    def pack(tensor):
        if hasattr(tensor,'ds_id'):
            if tensor.numel()!=tensor.ds_numel:
                raise RuntimeError('Diagnostic saved an unavailable ZeRO parameter')
            key=tensor.ds_id
            if key not in saved_parameters:saved_parameters[key]=tensor.detach().clone()
            return saved_parameters[key]
        # Saving the original Tensor here retains its grad_fn and creates
        # reference cycles on branches not traversed by the final objective.
        # Keep identical tensor values/storage, without the owning graph.
        return tensor.detach()
    try:
        with diagnostic_rng(),torch.enable_grad(),torch.autograd.graph.saved_tensors_hooks(pack,lambda value:value):
            output=module(*args)
            loss=objective(output)
            objectives=loss if isinstance(loss,dict) else {'value':loss}
            result={}
            for i,(name,value) in enumerate(objectives.items()):
                gradients=torch.autograd.grad(value,masks,allow_unused=False,retain_graph=i<len(objectives)-1)
                if any(not torch.isfinite(g).all() for g in gradients):raise FloatingPointError('Nonfinite diagnostic gradient')
                result[name]=[g.detach().clone() for g in gradients]
        for p,old in zip(masks,before):
            if (old is None)!=(p.grad is None) or (old is not None and not torch.equal(p.grad,old)):
                raise RuntimeError('Diagnostic pass changed accumulated mask gradients')
        return result if isinstance(loss,dict) else result['value']
    finally:
        for p,flag,(gradient,version) in zip(parameters,flags,parameter_gradients):
            if p.requires_grad!=flag:
                raise RuntimeError('Diagnostic pass changed video trainability')
            if p.grad is not gradient or (gradient is not None and gradient._version!=version):
                raise RuntimeError('Diagnostic pass changed accumulated video gradients')
