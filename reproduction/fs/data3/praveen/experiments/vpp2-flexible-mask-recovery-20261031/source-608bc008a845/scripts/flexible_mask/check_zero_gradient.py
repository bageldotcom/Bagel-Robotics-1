"""Small real ZeRO-3 regression before loading the full video model."""
import argparse,json,os,gc
from pathlib import Path
import torch
import torch.distributed as dist
import deepspeed
from torch.utils.checkpoint import checkpoint
from vpp2.flexible_mask.gradient_probe import frozen_weight_mask_gradient
from vpp2.flexible_mask.preflight import fingerprint
from vpp2.flexible_mask.runtime import distributed
from vpp2.flexible_mask.config import load,identity
from vpp2.compact_kv.spec import atomic_json

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.pre=torch.nn.Linear(32,32);self.layers=torch.nn.ModuleList([torch.nn.Linear(32,32) for _ in range(3)])
    def forward(self,x,mask):
        x=self.pre(x*mask.to(x.dtype)).tanh()
        for layer in self.layers:
            x=checkpoint(lambda value,layer=layer:layer(value*mask.to(value.dtype)).tanh(),x,use_reentrant=False)
        return x

def test(config,diagnostic):
    torch.manual_seed(123)
    ds=dict(train_batch_size=16,train_micro_batch_size_per_gpu=1,gradient_accumulation_steps=2,
        bf16=dict(enabled=True),data_types=dict(grad_accum_dtype='fp32'),communication_data_type='fp32',
        zero_allow_untested_optimizer=True,gradient_clipping=1.,steps_per_print=100000,
        zero_optimization=dict(stage=3,reduce_bucket_size=1024,stage3_prefetch_bucket_size=1024,
            stage3_param_persistence_threshold=0,stage3_max_live_parameters=1024,stage3_max_reuse_distance=0))
    with deepspeed.zero.Init(remote_device='cpu',dtype=torch.bfloat16,config_dict_or_path=ds):module=Tiny()
    optimizer=torch.optim.AdamW(module.parameters(),lr=1e-4,fused=True)
    engine,_,_,_=deepspeed.initialize(model=module,optimizer=optimizer,config=ds,dist_init_required=False)
    mask=torch.nn.Parameter(torch.ones(32,device='cuda'))
    mo=torch.optim.AdamW([mask],lr=1e-3)
    for update in range(3):
        for micro in range(2):
            gen=torch.Generator(device='cuda').manual_seed(200+update*2+micro+100*dist.get_rank())
            x=torch.randn((1,4,32),device='cuda',dtype=torch.bfloat16,generator=gen)
            engine._scale_wrt_gas=False
            if diagnostic:
                values=frozen_weight_mask_gradient(engine.module,[mask],lambda y:dict(gt=y.float().square().mean(),video=(y.float()-.2).square().mean(),kv=y.float().abs().mean()),(x,mask))
                assert all(torch.isfinite(v[0]).all() and v[0].norm()>0 for v in values.values())
            loss=engine(x,mask).float().square().mean()
            engine.backward(loss/2,scale_wrt_gas=False)
            boundary=engine.is_gradient_accumulation_boundary()
            if boundary:
                dist.all_reduce(mask.grad);mask.grad/=8
            engine.step()
            if boundary:mo.step();mo.zero_grad(set_to_none=True)
    state=fingerprint(dict(weights={n:p.ds_tensor for n,p in module.named_parameters()},optimizer=engine.optimizer.state_dict(),mask=mask,mask_optimizer=mo.state_dict()))
    engine.optimizer.destroy()
    del engine,module,optimizer,mo,mask;gc.collect();torch.cuda.empty_cache()
    return state

p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
c=load(a.config);rank,_,_=distributed(c)
baseline=test(c,False);measured=test(c,True)
assert baseline==measured,'Diagnostic changed small ZeRO-3 training state'
passed=torch.tensor(int(baseline==measured),device='cuda');dist.all_reduce(passed)
assert int(passed)==8
if rank==0:
    report=dict(identity=identity(c),passed=True,ranks=8,updates=3,accumulation=2,exact_optimizer_mask_and_weight_state=True)
    atomic_json(Path(c['root'])/'checks/zero_gradient_diagnostic.json',report)
    print(json.dumps(report),flush=True)
dist.barrier();dist.destroy_process_group()
