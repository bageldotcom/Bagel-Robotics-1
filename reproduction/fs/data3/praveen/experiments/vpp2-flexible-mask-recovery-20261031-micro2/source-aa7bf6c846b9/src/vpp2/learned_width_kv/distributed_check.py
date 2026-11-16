"""Exercise the installed DeepSpeed backward hooks against a direct FP32 oracle."""
import argparse
import copy
import json
from pathlib import Path
import torch
import torch.distributed as dist
import deepspeed
from deepspeed.utils import safe_get_full_grad
from vpp2.compact_kv.spec import atomic_json
from .runtime import initialize
from .common import identity,verify_receipt,kv_coefficient
from .reliability import kv_losses


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Tanh(),torch.nn.Linear(8,2))
    def forward(self,x,gate):return self.net(x*gate)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    config=json.loads(Path(a.config).read_text());rank,world,local,_=initialize(config)
    ident=identity(config);root=Path(config['root']);verify_receipt(root/'checks/cpu.json',ident)
    coefficient=kv_coefficient(config,'B',1)
    torch.manual_seed(124)
    base=Toy();state=copy.deepcopy(base.state_dict())
    x=torch.randn(32,4);target=torch.randn(32,2);g=torch.tensor([.3,.7,.5,.6])
    teacher=torch.randn(32,2)
    layer_offsets=torch.linspace(.01,.4,40).reshape(1,40,1)
    results=[]
    for case,weights in [('mixed',torch.tensor([0.]*16+[.5]*8+[1.]*8)),('all_rejected',torch.zeros(32))]:
        oracle=Toy();oracle.load_state_dict(state);og=torch.nn.Parameter(g.clone())
        prediction=oracle(x,og)
        action_teacher=((prediction-teacher).square().mean(-1)*weights).mean()
        terms=(prediction[:,None,:]-teacher[:,None,:]+layer_offsets).square()
        # Independent full-batch oracle: do not use the production reduction.
        loss=(prediction-target).square().mean()+action_teacher+coefficient*(terms*weights[:,None,None]).mean()+10*(og.sum()-2).square()
        loss.backward();expected=[q.grad.clone() for q in oracle.parameters()];expected_mask=og.grad.clone()
        for micro in (1,2,4):
            model=Toy();model.load_state_dict(state);model=model.to(f'cuda:{local}')
            opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
            ds=dict(train_batch_size=32,train_micro_batch_size_per_gpu=micro,gradient_accumulation_steps=4//micro,
                bf16=dict(enabled=False),fp16=dict(enabled=False),zero_optimization=dict(stage=3,overlap_comm=False),
                gradient_clipping=0.,communication_data_type='fp32',steps_per_print=1000000)
            engine,_,_,_=deepspeed.initialize(model=model,optimizer=opt,config=ds,dist_init_required=False)
            gate=torch.nn.Parameter(g.to(f'cuda:{local}'))
            gas=4//micro
            for j in range(gas):
                first=j*world*micro+rank*micro;batch=x[first:first+micro].to(f'cuda:{local}')
                labels=target[first:first+micro].to(f'cuda:{local}')
                teaching=teacher[first:first+micro].to(f'cuda:{local}')
                reliability=weights[first:first+micro].to(f'cuda:{local}')
                prediction=engine(batch,gate)
                terms=(prediction[:,None,:]-teaching[:,None,:]+layer_offsets.to(f'cuda:{local}')).square()
                _,kv,_=kv_losses(terms,reliability)
                action_teacher=((prediction-teaching).square().mean(-1)*reliability).mean()
                objective=(prediction-labels).square().mean()+action_teacher+coefficient*kv+10*(gate.sum()-2).square()
                engine.backward(objective/gas,scale_wrt_gas=False)
                if j==gas-1:
                    for parameter,want in zip(engine.module.parameters(),expected):
                        got=safe_get_full_grad(parameter).cpu()
                        torch.testing.assert_close(got,want,rtol=2e-6,atol=2e-7)
                    dist.all_reduce(gate.grad);gate.grad/=world
                    torch.testing.assert_close(gate.grad.cpu(),expected_mask,rtol=2e-6,atol=2e-7)
                engine.step()
            results.append(dict(case=case,micro_batch=micro,accumulation=gas,video_gradients_match_oracle=True,external_mask_gradients_match_oracle=True))
            del engine,model,opt,gate
            import gc
            gc.collect();torch.cuda.empty_cache()
    if rank==0:atomic_json(root/'checks/distributed.json',dict(passed=True,identity=ident,world_size=world,results=results,
        complete_loss_scaled_once=True,regularizer_not_multiplied_by_accumulation=True,
        teacher_supervision_gate='shared_action_and_kv_v1',kv_weight=coefficient,all_rejected_and_mixed_batches_verified=True))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
