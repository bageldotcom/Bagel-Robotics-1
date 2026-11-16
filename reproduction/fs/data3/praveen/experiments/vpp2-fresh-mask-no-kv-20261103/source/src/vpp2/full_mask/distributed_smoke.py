"""Real eight-GPU calibration, coverage probe, and optimizer/resume equivalence."""
import argparse
import copy
import json
from pathlib import Path
import time
import torch
import torch.distributed as dist
from .panel import immutable,BalancedStream,noise_stratum
from .train import identity,rng_state,restore_rng
from .runtime import distributed,initialize,install,model_versions,check_frozen,step_loss,calibrate_teacher,barrier,gather,verify_checkpoint_receipt
from .data import collate,prepared
from .curation import load_panel
from vpp2.learned_width_kv.controller import synchronize_controller


def assert_equal(left,right):
    if torch.is_tensor(left):
        if not torch.equal(left,right):raise AssertionError('Tensor differs after exact resume')
    elif isinstance(left,dict):
        if left.keys()!=right.keys():raise AssertionError('Resume keys differ')
        for key in left:assert_equal(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        if len(left)!=len(right):raise AssertionError('Resume lengths differ')
        for a,b in zip(left,right):assert_equal(a,b)
    elif left!=right:raise AssertionError('Resume scalar differs')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    config=json.loads(Path(args.config).read_text());panel=load_panel(config['panel_file']);rank,world=distributed()
    if world!=8:raise ValueError('The preflight requires exactly eight allocated GPUs')
    ident=identity(config,panel);gpu=json.loads(Path(config['gpu_receipt_file']).read_text())
    if gpu.get('experiment_identity')!=ident or not gpu.get('passed'):raise ValueError('One-GPU preflight does not match')
    verify_checkpoint_receipt(config,gpu,rank)
    model,reader,controller=initialize(config);branch=install(model);versions=model_versions(model)
    from .curation_v3_probe import probe
    probe(config,panel,8,2,initialized=(model,reader,controller,branch),rank=rank,world=world)
    thresholds=calibrate_teacher(config,model,branch,reader,panel,rank,world)
    cfg=config['train'];stream=BalancedStream([r for r in panel['windows'] if r['role']=='selection_initial'],config['seed'])
    optimizer=torch.optim.AdamW(controller.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['betas']),weight_decay=0.)
    def update(step):
        begin=time.monotonic();optimizer.zero_grad(set_to_none=True);totals=None
        for micro in range(cfg['global_batch']//world):
            position=step*cfg['global_batch']+micro*world+rank
            row=stream.at(position);stratum=noise_stratum(config['seed'],position,len(panel['tasks']))
            parts=prepared(model,collate(reader,[row],[position]),config['seed'],[stratum])
            loss,terms=step_loss(model,branch,controller,parts,[row],[stratum],thresholds,step,config)
            (loss/(cfg['global_batch']//world)).backward()
            totals=terms.detach().clone() if totals is None else totals+terms.detach()
            del loss,terms,parts
        norm=synchronize_controller(controller,cfg['gradient_clip']);optimizer.step();check_frozen(model,versions)
        dist.all_reduce(totals);totals/=cfg['global_batch'];torch.cuda.synchronize()
        if not torch.isfinite(totals).all():raise FloatingPointError('Nonfinite full-batch smoke metrics')
        return totals.detach().cpu(),float(norm),time.monotonic()-begin
    first=update(0)
    saved_controller=copy.deepcopy(controller.state_dict());saved_optimizer=copy.deepcopy(optimizer.state_dict());saved_rng=rng_state()
    uninterrupted=update(1);expected=copy.deepcopy(controller.state_dict());expected_optimizer=copy.deepcopy(optimizer.state_dict())
    controller.load_state_dict(saved_controller);optimizer.load_state_dict(saved_optimizer);restore_rng(saved_rng)
    resumed=update(1)
    assert_equal(controller.state_dict(),expected);assert_equal(optimizer.state_dict(),expected_optimizer);assert_equal(uninterrupted[0],resumed[0])
    # All replicas must contain identical selected scores after all-reduced updates.
    for parameter in controller.parameters():
        first_rank=parameter.detach().clone();dist.broadcast(first_rank,src=0)
        if not torch.equal(parameter,first_rank):raise AssertionError('Mask replicas diverged')
    checks=gather(dict(rank=rank,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        first_update_seconds=first[2],second_update_seconds=uninterrupted[2],replayed_update_seconds=resumed[2]))
    if rank==0:
        immutable(Path(config['distributed_receipt_file']),dict(passed=True,experiment_identity=ident,panel_identity=panel['identity'],
            world_size=world,global_batch=cfg['global_batch'],micro_batch=1,resume_bitwise_equal=True,
            all_replicas_equal=True,frozen_weights_unchanged=True,mask_updates_discarded=True,
            fresh_main_optimizer=True,main_sample_cursor=0,checks=checks,
            gt_action_loss=float(uninterrupted[0][0]),kv_loss=float(uninterrupted[0][3]),policy_retention_established=False))
    barrier()

if __name__=='__main__':main()
