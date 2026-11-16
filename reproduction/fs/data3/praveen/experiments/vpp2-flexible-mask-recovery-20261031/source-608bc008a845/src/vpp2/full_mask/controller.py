"""FP32 masks, global residual width and layer-specific flexible FFN allocation."""
import math
import torch
from torch import nn


def rank_select(scores,count):
    if not 0<count<=scores.numel() or not torch.isfinite(scores).all():raise ValueError('Invalid selection')
    return torch.argsort(scores.detach().flatten(),descending=True,stable=True)[:count]


def allocate_ffn(scores,total,minimum):
    layers,width=scores.shape
    if not layers*minimum<=total<=layers*width:raise ValueError('Infeasible FFN budget')
    mask=torch.zeros_like(scores)
    for layer in range(layers):mask[layer,rank_select(scores[layer],minimum)]=1
    extra=total-layers*minimum
    if extra:
        eligible=scores.detach().clone().masked_fill(mask.bool(),-torch.inf)
        # -inf marks protected indices, which cannot be chosen again.
        indices=torch.argsort(eligible.flatten(),descending=True,stable=True)[:extra]
        mask.view(-1)[indices]=1
    return mask


class Controller(nn.Module):
    def __init__(self,config):
        super().__init__();self.c=config['mask'];self.seed=config['seed']
        gen=torch.Generator().manual_seed(self.seed)
        # No old six-task selection is inherited. Small tie-breaking noise only.
        self.hidden_scores=nn.Parameter(torch.full((5120,),1.5)+torch.randn(5120,generator=gen)*1e-3)
        self.ffn_scores=nn.Parameter(torch.full((39,13824),1.5)+torch.randn((39,13824),generator=gen)*1e-3)
    def gates(self,update,mode='soft'):
        if mode=='open':return None,None
        if mode=='binary':
            h=torch.zeros_like(self.hidden_scores);h[rank_select(self.hidden_scores,self.c['keep_hidden'])]=1
            return h,allocate_ffn(self.ffn_scores,self.c['keep_ffn_total'],self.c['minimum_ffn_per_layer'])
        if mode!='soft':raise ValueError(mode)
        alpha=min((update+1)/self.c['entry_updates'],1.)
        # Keep temperature fixed while measuring channel stability; increasing it
        # can manufacture tiny score gradients without stable channel evidence.
        return tuple((1-alpha)+alpha*torch.sigmoid(x) for x in (self.hidden_scores,self.ffn_scores))
    def regularizer(self,update,h,f):
        progress=min((update+1)/self.c['budget_ramp_updates'],1.)
        ht=5120+(self.c['keep_hidden']-5120)*progress
        ft=39*13824+(self.c['keep_ffn_total']-39*13824)*progress
        budget=.5*((h.sum()-ht)/ht).square()+.5*((f.sum()-ft)/ft).square()
        # A soft lower bound prevents the global budget from collapsing one layer.
        floor=((self.c['minimum_ffn_per_layer']-f.sum(-1)).clamp_min(0)/self.c['minimum_ffn_per_layer']).square().mean()
        entropy=lambda g:-(g*g.clamp_min(1e-7).log()+(1-g)*(1-g).clamp_min(1e-7).log()).mean()/math.log(2)
        ent=.5*(entropy(h)+entropy(f));disc=.5*((2*torch.minimum(h,1-h)).mean()+(2*torch.minimum(f,1-f)).mean())
        sharp=max(0.,min((update+1-self.c['budget_ramp_updates'])/self.c['budget_ramp_updates'],1.))
        loss=self.c['budget_weight']*(budget+floor)+self.c['entropy_weight']*sharp*ent+self.c['discreteness_weight']*sharp*disc
        return loss,dict(budget=budget.detach(),floor=floor.detach(),entropy=ent.detach(),discreteness=disc.detach())
    @torch.no_grad()
    def selection(self):
        h,f=self.gates(0,'binary')
        return dict(hidden_indices=h.nonzero().flatten().cpu().tolist(),
            ffn_indices=[row.nonzero().flatten().cpu().tolist() for row in f]+[list(range(13824))],
            terminal_ffn='Kept unchanged; it occurs after the final action-visible KV cache.',
            mask_parameters=sum(p.numel() for p in self.parameters()))


def churn(a,b):
    def turnover(x,y):return len(set(x)^set(y))/max(1,len(set(x))+len(set(y)))
    return dict(hidden=turnover(a['hidden_indices'],b['hidden_indices']),
                ffn=[turnover(x,y) for x,y in zip(a['ffn_indices'],b['ffn_indices'])])
