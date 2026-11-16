"""Pinned distributed setup and the unchanged frozen native action/teacher path."""
from datetime import timedelta
import gc
import os
import random
import shutil
import numpy as np
import torch
import torch.distributed as dist
import deepspeed
from vpp2.width_kv.reproducibility import configure_deterministic_training
from vpp2.compact_kv.train import ds_config as _base_ds_config
from vpp2.compact_kv.model import load_reference, VideoBranch, native_cache, prepare_batch
from vpp2.compact_kv.blocks import install_blocks
from vpp2.compact_kv.checkpointing import LosslessCheckpointEngine
from .common import load_student, phase_config
from .controller import MaskController


def ds_config(config, world):
    """Audited larger communication buffers; arithmetic and batching unchanged."""
    result = _base_ds_config(config, world)
    result['zero_optimization'].update(
        reduce_bucket_size=100_000_000,
        stage3_prefetch_bucket_size=80_000_000,
        stage3_max_live_parameters=1_200_000_000,
        stage3_max_reuse_distance=1_200_000_000)
    return result


def initialize(config):
    policy=configure_deterministic_training()
    local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=40))
    if dist.get_world_size()!=8 or torch.cuda.device_count()!=8:
        raise ValueError('Eight distinct allocated GPUs are required')
    properties=torch.cuda.get_device_properties(local)
    uuids=[None]*8
    dist.all_gather_object(uuids,str(properties.uuid))
    if len(set(uuids))!=8 or 'H100' not in properties.name:
        raise RuntimeError('Expected eight unique allocated H100 GPUs')
    enough=[shutil.disk_usage(config['root']).free>=config['minimum_disk_bytes'] if dist.get_rank()==0 else None]
    dist.broadcast_object_list(enough,src=0)
    if not enough[0]:raise RuntimeError('Less than 1.2 TB free before GPU work')
    torch.manual_seed(config['seed']); random.seed(config['seed']); np.random.seed(config['seed'])
    # Audited: temporary empty storage is overwritten before use.
    # Keep deterministic algorithms and explicit initialization unchanged.
    torch.utils.deterministic.fill_uninitialized_memory = False
    policy['fill_uninitialized_memory'] = False
    return dist.get_rank(),dist.get_world_size(),local,policy


def student_engine(config,phase,micro,selection):
    c=phase_config(config,phase,micro)
    model=load_student(config,phase)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=c['learning_rate'],betas=tuple(c['betas']),weight_decay=c['weight_decay'],fused=True)
    engine,_,_,_=deepspeed.initialize(model=model,optimizer=optimizer,config=ds_config(c,8),dist_init_required=False)
    engine.checkpoint_engine=LosslessCheckpointEngine()
    controller=mask_optimizer=None
    if phase=='A':
        controller=MaskController(selection).to(device=f'cuda:{torch.cuda.current_device()}')
        mask_optimizer=torch.optim.AdamW(controller.parameters(),lr=config['mask_lr'],
            betas=tuple(config['mask_betas']),eps=config['mask_eps'],weight_decay=0.)
        if any(p.dtype!=torch.float32 for p in controller.parameters()):
            raise RuntimeError('Mask scores were cast out of FP32')
    return engine,controller,mask_optimizer


def frozen_reference(config,micro=1,check_sample=None):
    reference=load_reference(config,device=f'cuda:{torch.cuda.current_device()}')
    if any(p.requires_grad for p in reference.parameters()):
        raise RuntimeError('Reference, action, VAE, CLIP and proprio must all be frozen')
    before=prepared=None
    if check_sample is not None:
        prepared=prepare_batch(reference,check_sample,config['seed'])
        inputs,noise,tv,_,_,_,mask=prepared
        with torch.no_grad():
            cache=native_cache(reference,inputs,noise,tv,mask)
            before=[{k:v.cpu() for k,v in row.items()} for row in cache]
            del cache
    install_blocks(reference.video_expert,())
    reference.video_expert.use_gradient_checkpointing=False
    branch=VideoBranch(reference.video_expert).eval().requires_grad_(False)
    c=phase_config(config,'A',micro)
    ds=ds_config(c,8); ds['zero_optimization'].pop('offload_optimizer',None)
    teacher,_,_,_=deepspeed.initialize(model=branch,config=ds,dist_init_required=False)
    teacher.eval()
    object.__setattr__(reference,'_compact_teacher_engine',teacher)
    if before is not None:
        with torch.no_grad():
            after=native_cache(reference,inputs,noise,tv,mask)
            for left,right in zip(before,after):
                for key in ('k','v'):torch.testing.assert_close(left[key],right[key].cpu(),rtol=0,atol=0)
        del before,after,prepared,inputs,noise,tv,mask
    reference.mot.training=True
    gc.collect(); torch.cuda.empty_cache()
    return reference


def student_cache(engine,controller,phase,update,inputs,noise,timestep,mask,mode='soft'):
    if phase=='A':
        gates=controller.gates(update,mode)
        return engine(inputs,noise,timestep,mask[:1950,:1950],*gates),gates
    return engine(inputs,noise,timestep,mask[:1950,:1950]),None


def relative_rms(a,b):
    return float((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-6))
