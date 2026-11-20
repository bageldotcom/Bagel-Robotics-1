"""Physical tensor accounting, shared by search and export (never soft gate sums)."""
from dataclasses import dataclass
import math
import torch
from bagel_robotics._internal.adaptive_mask.slicing import slice_tensor, validate_indices

@dataclass(frozen=True)
class Geometry:
    hidden: int = 5120
    ffn: int = 13824
    layers: int = 40
    heads: int = 40
    head_dim: int = 128
    pack: int = 128
    hidden_floor: int = 4480
    ffn_floor: int = 4096
    head_floor: int = 8

    def __post_init__(self):
        if min(self.hidden,self.ffn,self.layers-1,self.heads,self.head_dim,self.pack)>0:
            if (self.ffn%self.pack == 0 and self.ffn_floor%self.pack == 0 and
                0<self.ffn_floor<=self.ffn and 0<self.head_floor<=self.heads and
                0<self.hidden_floor<=self.hidden): return
        raise ValueError('Invalid structural search bounds')

    def full(self):
        return dict(schema='adaptive-vpp2-selection-v1',hidden_indices=list(range(self.hidden)),
                    ffn_indices=[list(range(self.ffn)) for _ in range(self.layers)],
                    cross_head_indices=[list(range(self.heads)) for _ in range(self.layers)])

    def validate(self, selection):
        if selection.get('schema')!='adaptive-vpp2-selection-v1':raise ValueError('Wrong selection schema')
        validate_indices(selection['hidden_indices'],self.hidden,len(selection['hidden_indices']))
        if len(selection['hidden_indices'])<self.hidden_floor or len(selection['hidden_indices'])%self.pack:raise ValueError('Hidden floor violated')
        for key,total,floor,pack in [('ffn_indices',self.ffn,self.ffn_floor,self.pack),('cross_head_indices',self.heads,self.head_floor,1)]:
            if len(selection[key])!=self.layers:raise ValueError('Layer count changed')
            for i,ids in enumerate(selection[key]):
                validate_indices(ids,total,len(ids))
                if len(ids)<floor or len(ids)%pack:raise ValueError('Invalid layer allocation')
                if i==self.layers-1 and ids!=list(range(total)):raise ValueError('Terminal internals must remain complete')


def slice_parameter(name, tensor, selection, geometry):
    value,rule=slice_tensor(name,tensor,selection['hidden_indices'],selection['ffn_indices'],geometry.hidden)
    if name.startswith('blocks.') and '.cross_attn.' in name:
        _,layer,_,projection,kind=name.split('.')
        heads=selection['cross_head_indices'][int(layer)]
        coords=torch.tensor([h*geometry.head_dim+d for h in heads for d in range(geometry.head_dim)],device=value.device,dtype=torch.long)
        if projection in ('q','k','v','k_img','v_img','norm_q','norm_k','norm_k_img'):
            value=value.index_select(0,coords)
        elif projection=='o' and kind=='weight':value=value.index_select(1,coords)
        elif projection!='o':raise ValueError('Unaccounted cross-attention tensor '+name)
        rule+='; retain complete image/text conditioning heads'
    return value,rule

class ParameterInventory:
    def __init__(self, shapes, geometry):
        self.shapes={k:tuple(v) for k,v in shapes.items()};self.geometry=geometry
        self.original=sum(math.prod(s) for s in self.shapes.values());self._base={}
    @classmethod
    def from_model(cls,model,geometry):return cls({k:v.shape for k,v in model.state_dict().items()},geometry)
    def count(self,selection):
        self.geometry.validate(selection)
        # Same slicer as export, on meta tensors. Allocates no checkpoint weights.
        return sum(slice_parameter(k,torch.empty(s,device='meta'),selection,self.geometry)[0].numel() for k,s in self.shapes.items())
    def full_at_hidden(self,h):
        if h not in self._base:
            selection=self.geometry.full();selection['hidden_indices']=list(range(h))
            self._base[h]=self.count(selection)
        return self._base[h]
    def cost(self,h,ffn_counts,head_counts):
        g=self.geometry
        if len(ffn_counts)!=g.layers or len(head_counts)!=g.layers:raise ValueError('Invalid counts')
        # Exact I2V costs: 2 FFN linears; shared Q/O plus text K/V and image K/V;
        # three Q/K RMSNorm vectors. Terminal widths are included unchanged.
        total=self.full_at_hidden(h)
        total-=sum(g.ffn-n for n in ffn_counts)*(2*h+1)
        total-=sum(g.heads-n for n in head_counts)*g.head_dim*(6*h+8)
        return int(total)
