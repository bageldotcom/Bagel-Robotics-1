"""Spend mask updates on uncertain decisions; never delete or freeze model weights."""
from collections import deque
import copy
import torch

class MaskPriority:
    def __init__(self,controller,checks=5,stable_scale=.1,boundary_fraction=.05,
                 probability_range=.02,rank_range=.02):
        if checks<2 or not 0<stable_scale<=1 or not 0<boundary_fraction<.5:
            raise ValueError('Invalid mask-priority settings')
        self.controller=controller;self.checks=checks;self.stable_scale=stable_scale
        self.boundary_fraction=boundary_fraction;self.probability_range=probability_range;self.rank_range=rank_range
        self.names=(['hidden_scores'] if controller.hidden_scores.requires_grad else [])+['ffn_scores','cross_scores']
        self.history=deque(maxlen=checks)
        self.scales={n:torch.ones_like(getattr(controller,n)) for n in self.names}
        self.previous=self._hard();self.changed={n:torch.zeros_like(x,dtype=torch.bool) for n,x in self.previous.items()}
    def _hard(self):
        gates=self.controller.gates(mode='binary')
        values=dict(hidden_scores=gates[0].detach().cpu().bool(),ffn_scores=gates[1].detach().cpu().bool(),cross_scores=gates[2].detach().cpu().bool())
        return {n:values[n] for n in self.names}
    @torch.no_grad()
    def refresh(self):
        """Any decision change between checks immediately regains full priority."""
        hard=self._hard()
        for n in self.names:
            change=hard[n]!=self.previous[n];self.changed[n]|=change
            self.scales[n].masked_fill_(change.to(self.scales[n].device),1.)
        self.previous=hard
    @torch.no_grad()
    def reopen(self):
        # A task-level quality warning invalidates confidence, not native GT loss.
        self.history.clear()
        for n in self.names:self.scales[n].fill_(1.);self.changed[n].zero_()
        self.previous=self._hard()
    @torch.no_grad()
    def observe(self,pruning_active=True):
        self.refresh()
        if not pruning_active:
            self.reopen();return self.summary()
        item={};hard=self._hard()
        for n in self.names:
            scores=getattr(self.controller,n).detach().cpu()
            order=torch.argsort(scores,dim=-1,descending=True,stable=True)
            ranks=torch.empty_like(scores)
            places=torch.arange(scores.shape[-1],dtype=scores.dtype).expand_as(scores)
            ranks.scatter_(-1,order,places)
            cut=hard[n].sum(-1,keepdim=True).float()
            informative=(cut>0)&(cut<scores.shape[-1])
            away=((ranks+.5-cut).abs()/scores.shape[-1])>=self.boundary_fraction
            item[n]=dict(hard=hard[n],probability=scores.sigmoid(),rank=ranks/max(scores.shape[-1]-1,1),
                eligible=informative&away,changed=self.changed[n].clone())
            self.changed[n].zero_()
        self.history.append(item)
        for n in self.names:
            self.scales[n].fill_(1.)
            if len(self.history)<self.checks:continue
            records=[x[n] for x in self.history]
            decisions=torch.stack([x['hard'] for x in records])
            probability=torch.stack([x['probability'] for x in records]);ranks=torch.stack([x['rank'] for x in records])
            certain=(decisions==decisions[0]).all(0)
            certain&=torch.stack([x['eligible'] for x in records]).all(0)
            certain&=~torch.stack([x['changed'] for x in records]).any(0)
            certain&=(probability.max(0).values-probability.min(0).values)<=self.probability_range
            certain&=(ranks.max(0).values-ranks.min(0).values)<=self.rank_range
            self.scales[n].masked_fill_(certain.to(self.scales[n].device),self.stable_scale)
        return self.summary()
    @torch.no_grad()
    def step(self,optimizer):
        """Scale the actual Adam displacement, not a gradient Adam can renormalize."""
        params={id(p) for group in optimizer.param_groups for p in group['params']}
        expected={id(getattr(self.controller,n)) for n in self.names}
        if params!=expected:raise ValueError('Priority optimizer must contain exactly the enabled mask scores')
        before={n:getattr(self.controller,n).detach().clone() for n in self.names}
        optimizer.step()
        for n in self.names:
            p=getattr(self.controller,n);scale=self.scales[n].to(p)
            p.copy_(torch.where(scale==1,p,before[n]+(p-before[n])*scale))
    def summary(self):
        out={};hard=self._hard()
        for n in self.names:
            scales=self.scales[n].detach().cpu();certain=scales<1;keep=hard[n]
            label={'hidden_scores':'hidden','ffn_scores':'ffn','cross_scores':'cross_heads'}[n]
            for k,v in dict(uncertain_fraction=(~certain).float().mean(),stable_keep_fraction=(certain&keep).float().mean(),
                           stable_off_fraction=(certain&~keep).float().mean(),mean_update_scale=scales.mean()).items():
                out[f'priority/{label}/{k}']=float(v)
        return out
    def state_dict(self):
        return dict(history=copy.deepcopy(list(self.history)),scales={k:v.detach().cpu().clone() for k,v in self.scales.items()},
                    previous=copy.deepcopy(self.previous),changed=copy.deepcopy(self.changed))
    def load_state_dict(self,state):
        self.history=deque(copy.deepcopy(state['history']),maxlen=self.checks)
        self.previous=copy.deepcopy(state['previous']);self.changed=copy.deepcopy(state['changed'])
        for n in self.names:self.scales[n].copy_(state['scales'][n].to(self.scales[n]))
