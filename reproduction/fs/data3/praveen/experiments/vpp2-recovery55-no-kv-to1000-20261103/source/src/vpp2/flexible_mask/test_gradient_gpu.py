"""GPU check: objective probes do not change the next ZeRO-3 update.

Uses the actual installed DeepSpeed, BF16, checkpoint recomputation, and all
eight allocated ranks. It is run before loading the real model.
"""
import argparse
import copy
import json
from pathlib import Path
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import deepspeed


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks=torch.nn.ModuleList([torch.nn.Linear(16,32),torch.nn.Linear(32,16)])
        self.head=torch.nn.Linear(16,4)

    def forward(self,x):
        from torch.utils.checkpoint import checkpoint
        def block(y):return self.blocks[1](torch.nn.functional.gelu(self.blocks[0](y)))
        h=checkpoint(block,x,use_reentrant=False)
        return self.head(h),h


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    from .config import load,identity
    from .runtime import distributed,ds_config
    from .preflight import fingerprint
    from .recovery_gradients import measured_gradients,clear_gradients,engine_counters,adam_state,sample_indices,sample_parameters
    from .rng import rng_state
    from vpp2.compact_kv.spec import atomic_json
    c=load(a.config);rank,world,_=distributed(c);device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    # This size reproduces the original float32 endpoint-rounding failure.
    length=18_677_760
    old=torch.linspace(0,length-1,64,device=device,dtype=torch.float32).long()
    if old[-1].item()<length:raise RuntimeError('Regression case no longer exercises the original issue')
    indices=sample_indices(length,device)
    vector=torch.zeros(length,device=device,dtype=torch.float32);vector[-1]=7.
    sampled=vector[indices];torch.cuda.synchronize()
    if indices[-1].item()!=length-1 or sampled[-1].item()!=7.:raise RuntimeError('Invalid large-shard diagnostic indexing')
    del vector,sampled,indices,old
    torch.manual_seed(72001);template=Toy().state_dict();engines=[]
    for _ in range(2):
        model=Toy();model.load_state_dict(template);model=model.to(device,dtype=torch.bfloat16)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-6,betas=(.9,.999),weight_decay=.01,fused=True)
        ds=ds_config(c);ds['train_micro_batch_size_per_gpu']=1;ds['gradient_accumulation_steps']=2;ds['train_batch_size']=2*world
        eng,_,_,_=deepspeed.initialize(model=model,optimizer=opt,config=ds,dist_init_required=False)
        engines.append(eng)
    def make(eng,step):
        g=torch.Generator(device='cpu').manual_seed(10000+step*world+rank)
        x=torch.randn(1,16,generator=g).to(device,dtype=torch.bfloat16)
        target=torch.randn(1,4,generator=g).to(device)
        eng._scale_wrt_gas=False
        action,hidden=eng(x)
        return {'gt':(action.float()-target).square().mean(),
                'video':.05*hidden.float().square().mean(),
                'kv':hidden.float().sum()*0.}
    def update(eng,start):
        for step in range(start,start+2):
            values=make(eng,step)
            eng.backward(sum(values.values())/2,scale_wrt_gas=False);eng.step()
    for eng in engines:update(eng,0)
    before_rng=fingerprint(rng_state());before_opt=fingerprint(adam_state(engines[1]))
    before_model=fingerprint([p.ds_tensor for p in engines[1].module.parameters()])
    before_steps=(engines[1].global_steps,engines[1].micro_steps)
    sample_parameters(SimpleNamespace(engine=engines[1]))
    def terms(i):return make(engines[1],i),True
    measured=measured_gradients(engines[1],terms,[10,11],world,2*world,device,active_names=('gt','video'))
    if before_rng!=fingerprint(rng_state()) or before_opt!=fingerprint(adam_state(engines[1])):
        raise RuntimeError('Diagnostic changed random/Adam state')
    if before_model!=fingerprint([p.ds_tensor for p in engines[1].module.parameters()]):
        raise RuntimeError('Diagnostic changed model weights')
    if before_steps!=(engines[1].global_steps,engines[1].micro_steps):raise RuntimeError('Changed engine counters')
    for eng in engines:update(eng,2)
    fp=[fingerprint(dict(weights=[p.ds_tensor for p in e.module.parameters()],
                         master=e.optimizer.fp32_partitioned_groups_flat,
                         adam=adam_state(e),
                         counters=(e.global_steps,e.micro_steps))) for e in engines]
    okay=torch.tensor(int(fp[0]==fp[1]),device=device);dist.all_reduce(okay,op=dist.ReduceOp.MIN)
    if not okay.item():raise RuntimeError('Next optimizer update differs after read-only diagnostic')
    if not all(measured['all'][k+'_norm']>0 for k in ['gt','video']):
        raise RuntimeError('Objective gradient is missing')
    if measured['all']['kv_norm']!=0.:raise RuntimeError('Disabled K/V has a gradient')
    if rank==0:
        atomic_json(Path(c['root'])/'checks/gradient_noninterference_gpu.json',dict(
            identity=identity(c),passed=True,world_size=world,zero_stage=3,bf16=True,
            checkpointed_blocks=True,next_update_bitwise_equal=True,adam_unchanged_by_diagnostic=True,
            rng_unchanged=True,large_shard_integer_indices_verified=True,parameter_sampling_noninterference=True,scope='Small GPU model; real-model fingerprints checked separately at each probe'))
    dist.barrier()


if __name__=='__main__':main()
