"""Hard selections in every forward; differentiable score surrogate in backward."""
import copy
import torch
import torch.distributed as dist
from torch import nn
from .geometry import Geometry

def ranking(scores):
    if not torch.isfinite(scores).all():raise FloatingPointError('Nonfinite mask score')
    return torch.argsort(scores.detach().flatten(),descending=True,stable=True).cpu().tolist()

def straight_through(scores,hard):
    soft=scores.sigmoid() # fixed temperature one; no artificial saturation schedule
    return hard+(soft-soft.detach())

class Controller(nn.Module):
    def __init__(self,geometry=Geometry(),seed=100008,learn_hidden=True):
        super().__init__();self.geometry=geometry
        gen=torch.Generator().manual_seed(seed)
        self.hidden_scores=nn.Parameter(torch.randn(geometry.hidden,generator=gen)*1e-3,requires_grad=learn_hidden)
        self.ffn_scores=nn.Parameter(torch.randn(geometry.layers-1,geometry.ffn,generator=gen)*1e-3)
        self.cross_scores=nn.Parameter(torch.randn(geometry.layers-1,geometry.heads,generator=gen)*1e-3)
        self._selection=geometry.full()
        # Execution cache only; never serialize it or reuse a differentiable graph.
        self.cache_hard_masks=False
        self._hard_mask_cache=None
    def selection(self):return copy.deepcopy(self._selection)
    def set_selection(self,value):
        self.geometry.validate(value)
        if not self.hidden_scores.requires_grad and len(value["hidden_indices"])!=self.geometry.hidden:raise ValueError("Shared hidden width is protected in this pilot")
        self._selection=copy.deepcopy(value)
        self._hard_mask_cache=None
    def gates(self,update=0,mode='train'):
        if mode not in ('train','binary'):raise ValueError(mode)
        selection=self._selection;g=self.geometry
        values=[]
        params=(self.hidden_scores,self.ffn_scores,self.cross_scores)
        signature=tuple((p.device,p.dtype,tuple(p.shape)) for p in params)
        cached=self._hard_mask_cache if self.cache_hard_masks and mode=='train' else None
        if cached is not None and cached[0]!=signature:cached=None
        hard_values=[]
        for index,(scores,key) in enumerate([(self.hidden_scores,'hidden_indices'),(self.ffn_scores,'ffn_indices'),(self.cross_scores,'cross_head_indices')]):
            if cached is not None:hard=cached[1][index]
            else:
                hard=torch.zeros_like(scores)
                if hard.ndim==1:hard[selection[key]]=1
                else:
                    for i,ids in enumerate(selection[key][:-1]):hard[i,ids]=1
            hard_values.append(hard)
            values.append(straight_through(scores,hard) if mode=='train' else hard)
        if self.cache_hard_masks and mode=='train' and cached is None:
            self._hard_mask_cache=(signature,tuple(hard_values))
        return tuple(values)
    def forward_gates(self,mode='train'):
        gates=self.gates(mode=mode)
        # Protected residual width uses native normalization directly.
        return gates if self.hidden_scores.requires_grad else (None,gates[1],gates[2])
    @torch.no_grad()
    def project(self,inventory,cap,hidden_count):
        g=self.geometry
        if not g.hidden_floor<=hidden_count<=g.hidden:raise ValueError('Hidden count out of bounds')
        hids=sorted(ranking(self.hidden_scores)[:hidden_count])
        f_orders=[ranking(x) for x in self.ffn_scores]
        c_orders=[ranking(x) for x in self.cross_scores]
        fcounts=[g.ffn_floor]*(g.layers-1)+[g.ffn]
        ccounts=[g.head_floor]*(g.layers-1)+[g.heads]
        used=inventory.cost(hidden_count,fcounts,ccounts)
        if used>cap:raise ValueError('Budget cannot accommodate protected floors')
        candidates=[]
        fprob=self.ffn_scores.sigmoid().cpu();cprob=self.cross_scores.sigmoid().cpu()
        for layer in range(g.layers-1):
            for start in range(g.ffn_floor,g.ffn,g.pack):
                ids=f_orders[layer][start:start+g.pack]
                candidates.append((float(fprob[layer,ids].mean()),0,layer,start,g.pack*(2*hidden_count+1)))
            for start in range(g.head_floor,g.heads):
                candidates.append((float(cprob[layer,c_orders[layer][start]]),1,layer,start,g.head_dim*(6*hidden_count+8)))
        # Identical probability ranking at train, monitoring and export; no
        # posthoc score/cost ratio. Stable family/layer/position tie breaking.
        candidates.sort(key=lambda x:(-x[0],x[1],x[2],x[3]))
        for _,family,layer,start,cost in candidates:
            counts=fcounts if family==0 else ccounts
            if start!=counts[layer]:continue # preserve prefix ordering
            if used+cost<=cap:
                counts[layer]+=g.pack if family==0 else 1;used+=cost
        selected=dict(schema='adaptive-vpp2-selection-v1',hidden_indices=hids,
            ffn_indices=[sorted(order[:count]) for order,count in zip(f_orders,fcounts)]+[list(range(g.ffn))],
            cross_head_indices=[sorted(order[:count]) for order,count in zip(c_orders,ccounts)]+[list(range(g.heads))])
        g.validate(selected)
        if inventory.cost(hidden_count,fcounts,ccounts)!=used:raise AssertionError('Physical cost inconsistency')
        self.set_selection(selected)
        return dict(parameters=used,cap=int(cap),hidden=hidden_count,ffn=fcounts,cross_heads=ccounts)
    def trainable_parameters(self):return [p for p in self.parameters() if p.requires_grad]
    def get_extra_state(self):return self.selection()
    def set_extra_state(self,state):self.set_selection(state)


def synchronize_controller(controller,max_norm=1.):
    """Average exactly once after accumulation, then clip the global gradient."""
    params=controller.trainable_parameters()
    for parameter in params:
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError('Missing or nonfinite trainable-mask gradient')
        if dist.is_initialized():
            dist.all_reduce(parameter.grad,op=dist.ReduceOp.SUM)
            parameter.grad.div_(dist.get_world_size())
    for parameter in controller.parameters():
        if not parameter.requires_grad and parameter.grad is not None:
            raise AssertionError('Protected mask received a gradient')
    return torch.nn.utils.clip_grad_norm_(params,max_norm,error_if_nonfinite=True)
