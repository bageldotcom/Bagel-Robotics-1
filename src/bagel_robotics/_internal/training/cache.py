"""Bounded CPU cache of frozen encoder outputs; no noise or model K/V."""
from collections import OrderedDict
import copy
import torch

class RecoveryMemoryCache:
    """Bounded lossless frozen-conditioning cache; no persistent window expansion."""
    def __init__(self,capacity=128):
        if type(capacity) is not int or capacity<8:raise ValueError('Cache must fit a complete microbatch')
        self.capacity=capacity;self.entries=OrderedDict()
    def load(self,key):
        if key not in self.entries:return None
        self.entries.move_to_end(key)
        return {k:(v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k,v in self.entries[key].items()}
    def save(self,key,inputs):
        payload={k:(v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k,v in inputs.items()}
        payload['input_latents']=None
        self.entries[key]=payload;self.entries.move_to_end(key)
        while len(self.entries)>self.capacity:self.entries.popitem(last=False)
