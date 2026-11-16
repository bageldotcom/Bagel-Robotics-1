"""Eight-rank BF16 ZeRO-3 checkpoint test with the real save/load functions."""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
import deepspeed


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    from .config import load,identity
    from .runtime import distributed,ds_config
    from .test_gradient_gpu import Toy
    from .recovery_gradients import adam_state
    from .preflight import fingerprint
    from .rng import rng_state,restore_rng
    from . import checkpoint
    from vpp2.compact_kv.spec import atomic_json
    config=load(args.config);rank,world,_=distributed(config)
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.manual_seed(81931);template=Toy().state_dict()
    def engine():
        model=Toy();model.load_state_dict(template);model=model.to(device,dtype=torch.bfloat16)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-6,betas=(.9,.999),weight_decay=.01,fused=True)
        ds=ds_config(config);ds['train_micro_batch_size_per_gpu']=1
        ds['gradient_accumulation_steps']=2;ds['train_batch_size']=2*world
        value,_,_,_=deepspeed.initialize(model=model,optimizer=opt,config=ds,dist_init_required=False)
        return value
    baseline=engine()
    def update(eng):
        for _ in range(2):
            # Both global CPU and device RNG are consumed to expose lost state.
            x=torch.randn(1,16).to(device,dtype=torch.bfloat16)
            target=torch.randn(1,4,device=device)
            eng._scale_wrt_gas=False
            predicted,hidden=eng(x)
            loss=(predicted.float()-target).square().mean()+.05*hidden.float().square().mean()
            eng.backward(loss/2,scale_wrt_gas=False);eng.step()
    def payload(eng):
        return dict(weights=[p.ds_tensor for p in eng.module.parameters()],
                    master=eng.optimizer.fp32_partitioned_groups_flat,adam=adam_state(eng),
                    global_steps=eng.global_steps,rng=rng_state())
    update(baseline)
    root=Path(config['root'])/'checks/resume_gpu_artifacts'
    rt=SimpleNamespace(rank=rank,world=world,engine=baseline,masks=None,mask_optimizer=None,
                       recovery_export_sha256='small-resume-test')
    checkpoint.save(rt,root,dict(identity=identity(config),update=1,stage='recovery',best=[],
                                recovery_subset_cursor=256))
    saved_rng=rng_state();before=fingerprint(payload(baseline))
    resumed=engine();rr=SimpleNamespace(rank=rank,world=world,engine=resumed,masks=None,
                                      mask_optimizer=None,recovery_export_sha256='small-resume-test')
    state=checkpoint.load(rr,root,identity(config),'recovery',tag='update_000001')
    if state['cursor']!=256 or state['recovery_subset_cursor']!=256:
        raise RuntimeError('Checkpoint data cursor was lost')
    restored=fingerprint(payload(resumed))
    if before!=restored:raise RuntimeError('Weights, optimizer, counter or RNG differ after restore')
    restore_rng(saved_rng);update(baseline);expected=fingerprint(payload(baseline))
    restore_rng(saved_rng);update(resumed);actual=fingerprint(payload(resumed))
    okay=torch.tensor(int(expected==actual),device=device);dist.all_reduce(okay,op=dist.ReduceOp.MIN)
    if not okay.item():raise RuntimeError('First optimizer update after checkpoint restore differs')
    if rank==0:
        atomic_json(Path(config['root'])/'checks/recovery_resume_gpu.json',dict(
            identity=identity(config),passed=True,world_size=world,zero_stage=3,bf16=True,
            model_restored_exactly=True,adam_restored_exactly=True,rng_restored_exactly=True,
            data_cursor_preserved=True,next_update_bitwise_equal=True,
            scope='Small model with actual distributed checkpoint I/O; real checkpoint250 predictions verified separately'))
    dist.barrier()


if __name__=='__main__':main()
