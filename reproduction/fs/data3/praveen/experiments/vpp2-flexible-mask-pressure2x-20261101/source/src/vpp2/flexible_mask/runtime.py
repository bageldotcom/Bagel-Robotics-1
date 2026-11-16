"""Separate teacher/student storage and explicit ZeRO-3 student construction."""
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.distributed as dist
from vpp2.adaptive_mask.geometry import Geometry, ParameterInventory
from vpp2.adaptive_mask.masking import install
from vpp2.adaptive_mask.data import future_mask
from .data import prepare_cached as prepare,reader_configuration
from vpp2.compact_kv.model import load_reference, native_config, action_prediction
from vpp2.full_mask.data import WindowReader
from .controller import Masks, SoftCost, schedule
from .losses import objectives, cache_terms


def distributed(config):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU work must run inside a Slurm allocation')
    local=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    # Explicit opt-in; verify all buffers are initialized before adopting it.
    torch.utils.deterministic.fill_uninitialized_memory=config.get('execution',{}).get('fill_uninitialized_memory',True)
    if not dist.is_initialized():dist.init_process_group('nccl',timeout=timedelta(minutes=40))
    if dist.get_world_size()!=8:raise ValueError('This recipe requires eight GPUs')
    attach_deepspeed_backend()
    uuids=[None]*8
    dist.all_gather_object(uuids,str(torch.cuda.get_device_properties(local).uuid))
    if len(set(uuids))!=8:raise RuntimeError('Multiple workers share a physical GPU')
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    return dist.get_rank(),dist.get_world_size(),local


def attach_deepspeed_backend():
    """Register the existing process group before ZeRO reads its batch config."""
    import deepspeed
    if not dist.is_initialized():raise RuntimeError('PyTorch process group is not initialized')
    deepspeed.init_distributed(dist_backend=dist.get_backend(),dist_init_required=False)
    if (deepspeed.comm.get_world_size(),deepspeed.comm.get_rank())!=(dist.get_world_size(),dist.get_rank()):
        raise RuntimeError('DeepSpeed and PyTorch process groups disagree')


def ds_config(config,training=True):
    t=config['train']
    result=dict(train_batch_size=t['global_batch'],train_micro_batch_size_per_gpu=t['micro_batch'],
        gradient_accumulation_steps=t['global_batch']//(8*t['micro_batch']),
        bf16=dict(enabled=True),data_types=dict(grad_accum_dtype='fp32'),communication_data_type='fp32',gradient_clipping=1.,
        zero_allow_untested_optimizer=True,steps_per_print=1000000,
        zero_optimization=dict(stage=3,overlap_comm=True,contiguous_gradients=True,
            reduce_bucket_size=100_000_000,stage3_prefetch_bucket_size=80_000_000,
            stage3_param_persistence_threshold=100_000,stage3_max_live_parameters=1_200_000_000,
            stage3_max_reuse_distance=1_200_000_000,stage3_gather_16bit_weights_on_model_save=True))
    return result


def populate_partitioned(model,state):
    import deepspeed
    named=dict(model.named_parameters())
    if set(named)!=set(state):raise ValueError('Source tensor inventory mismatch')
    for name,p in named.items():
        shape=tuple(getattr(p,'ds_shape',p.shape))
        if shape!=tuple(state[name].shape):raise ValueError('Source tensor shape mismatch: '+name)
        with deepspeed.zero.GatheredParameters([p],modifier_rank=0):
            if dist.get_rank()==0:
                with torch.no_grad():p.copy_(state[name].to(device=p.device,dtype=p.dtype))


def independent_storage(student,teacher):
    def storage(model):
        return {(str(p.device),p.untyped_storage().data_ptr()) for q in model.parameters()
                for p in [getattr(q,'ds_tensor',q)] if p.numel()}
    if storage(student)&storage(teacher):raise RuntimeError('Teacher/student writable storage alias')


class FrozenReplica(torch.nn.Module):
    """Frozen resident teacher; same branch interface without ZeRO fetch hooks."""
    def __init__(self,module):
        super().__init__()
        if any(p.requires_grad for p in module.parameters()):
            raise ValueError('Resident teacher must already be frozen')
        self.module=module
        self.eval()

    def forward(self,*args,**kwargs):
        return self.module(*args,**kwargs)


class Runtime:
    def __init__(self,config,recovery_export=None):
        import deepspeed
        from omegaconf import OmegaConf
        from vpp2.models.wan21_14b.wan_video_dit import WanVideoDiT,precompute_freqs_cis_3d
        from vpp2.compact_kv.checkpointing import LosslessCheckpointEngine
        from .export import read_export_manifest, export_state
        from vpp2.full_mask.panel import file_hash
        self.recovery_export_sha256=None if recovery_export is None else file_hash(Path(recovery_export)/'manifest.json')
        self.physical_parameters=config['original_video_parameters']
        self.config=config;self.rank=dist.get_rank();self.world=dist.get_world_size()
        self.device=torch.device('cuda',torch.cuda.current_device())
        self.geometry=Geometry(**config['geometry'])
        inputs=json.loads((Path(config['root'])/'inputs.json').read_text())
        self.inventory=ParameterInventory(inputs['shapes'],self.geometry)
        self.cost=SoftCost(self.inventory)
        self.kwargs=OmegaConf.to_container(native_config(config).model.video_dit_config,resolve=True)
        # Construct BF16 shards directly. Never materialize a full FP32 GPU student.
        ds=ds_config(config)
        with deepspeed.zero.Init(remote_device='cpu',pin_memory=True,dtype=torch.bfloat16,config_dict_or_path=ds):
            if recovery_export:
                from vpp2.adaptive_mask.export import build_compact
                manifest=read_export_manifest(recovery_export,require_verified=True)
                self.physical_parameters=manifest['parameters']
                video=build_compact(manifest['model_kwargs'],manifest['selection'],self.geometry,device='cpu')
            else:
                video=WanVideoDiT(**self.kwargs)
        state=(export_state(recovery_export) if recovery_export else
               torch.load(Path(config['teacher_bundle'])/'video.pt',map_location='cpu',mmap=True,weights_only=True)['dit'])
        populate_partitioned(video,state);del state
        video.freqs=precompute_freqs_cis_3d(self.geometry.head_dim)
        video.requires_grad_(True)  # Includes final video-producing block and head.
        video.use_gradient_checkpointing=True
        video.activation_checkpoint_layers=frozenset(range(40))
        video.cache_mask_open_flags=config.get('execution',{}).get('cached_norm_flags',False)
        branch=install(video).train()
        lr=config['recovery']['learning_rate'] if recovery_export else config['train']['video_lr']
        optimizer=torch.optim.AdamW(branch.parameters(),lr=lr,betas=(.9,.999),weight_decay=.01,fused=True)
        self.engine,_,_,_=deepspeed.initialize(model=branch,optimizer=optimizer,config=ds,dist_init_required=False)
        self.engine.checkpoint_engine=LosslessCheckpointEngine()
        self.masks=None;self.mask_optimizer=None
        if not recovery_export:
            rankings=torch.load(config['initial_masks'],map_location='cpu',weights_only=False)['state']['controller']
            self.masks=Masks(self.geometry,rankings,config['initial_preferred']).to(self.device)
            self.mask_optimizer=torch.optim.AdamW(self.masks.parameters(),lr=.01,betas=(.9,.999),weight_decay=0.)
        # Load the frozen native policy on CPU, then shard its video and move only
        # action/auxiliary modules. model.to(cuda) would re-materialize its video.
        previous_meta=os.environ.get('VPP2_LOAD_META')
        try:
            os.environ['VPP2_LOAD_META']='1'
            self.reference=load_reference(config,device='cpu')
        finally:
            if previous_meta is None:os.environ.pop('VPP2_LOAD_META',None)
            else:os.environ['VPP2_LOAD_META']=previous_meta
        ref=self.reference;ref.device=self.device
        for name in ('action_expert','proprio_encoder','clip_encoder','vae','text_encoder'):
            module=getattr(ref,name,None)
            if module is not None:module.to(self.device)
        ref._restore_auxiliary_component_dtypes()
        ref.video_expert.use_gradient_checkpointing=False
        teacher=install(ref.video_expert).eval().requires_grad_(False)
        storage=config.get('execution',{}).get('teacher_storage','sharded')
        if storage=='sharded':
            # Partition preloaded BF16 CPU parameters before DeepSpeed's device move.
            with deepspeed.zero.Init(module=teacher,remote_device='cpu',pin_memory=True,
                                     dtype=torch.bfloat16,config_dict_or_path=ds):pass
            self.teacher,_,_,_=deepspeed.initialize(model=teacher,config=ds,dist_init_required=False)
        elif storage=='replicated':
            # Free unused initialization buffers before allocating the complete,
            # immutable teacher. Student sharding/optimizer are unchanged.
            torch.cuda.empty_cache()
            self.teacher=FrozenReplica(teacher.to(self.device))
        else:raise ValueError('Unknown frozen teacher storage mode')
        self.teacher.eval();ref.action_expert.use_gradient_checkpointing=True;ref.mot.training=True
        ref.mot.skip_nograd_checkpointing=config.get('execution',{}).get('skip_nograd_checkpointing',False)
        ref.mot.mot_checkpoint_mixed_attn=config.get('execution',{}).get('mixed_attention_checkpointing',True)
        self.reader=WindowReader(reader_configuration(config));ref._full_mask_conditioning_cache=self.reader.runtime_cache
        self.reader.reuse_prepared_payloads=config.get('execution',{}).get('deduplicate_conditioning_reads',False)
        independent_storage(self.engine.module,self.teacher.module)
        if any(p.requires_grad for p in ref.parameters()):raise RuntimeError('Reference must be frozen')
        if any(isinstance(m,torch.nn.Dropout) and m.p>0 for m in self.engine.module.modules()):
            raise RuntimeError('Unexpected stochastic dropout')

    def gates(self,update,mode='soft'):
        return (None,None,None) if self.masks is None else self.masks.gates(update,mode)

    def video(self,parts,teacher=False,gates=(None,None,None),return_video=True):
        engine=self.teacher if teacher else self.engine
        return engine(parts[0],parts[1],parts[2],parts[6][:1950,:1950],*gates,return_video=return_video)

    def predict(self,parts,cache):
        return action_prediction(self.reference,parts[0],cache,parts[3],parts[4],parts[6])

    def measure_gt_mask_gradients(self,parts,update,rows,teacher,teacher_action):
        from .gradient_probe import frozen_weight_mask_gradient
        from vpp2.compact_kv.losses import action_error
        def objective(outputs):
            prediction=self.predict(parts,outputs['cache'])
            _,_,_,terms=objectives(prediction,teacher_action,parts[5],parts[0]['action_loss_is_pad'],
                self.reference.train_action_scheduler.training_weight(parts[4]),outputs['video'],teacher['video'],
                cache_terms(outputs['cache'],teacher['cache']),future_mask(rows,self.device),schedule(update)['teacher'])
            return terms
        return frozen_weight_mask_gradient(self.engine.module,self.masks.families(),objective,
            (parts[0],parts[1],parts[2],parts[6][:1950,:1950],*self.gates(update)))

    def forward(self,rows,positions,update,mode='soft',measure_gt=False):
        from vpp2.full_mask.panel import seed_for
        # Set this before any optional module-only diagnostic backward.
        self.engine._scale_wrt_gas=False
        calibration_chunk=self.config.get('execution',{}).get('calibration_micro_batch',len(rows))
        split_calibration=measure_gt and calibration_chunk<len(rows)
        measured=None
        if split_calibration:
            from .calibration_batch import measure_in_chunks
            from .rng import diagnostic_rng
            def measure_small(small_rows,small_positions):
                small_strata=[seed_for(self.config['seed'],'stratum',p)%4 for p in small_positions]
                small_parts=prepare(self.reference,self.reader,small_rows,small_positions,self.config['seed'],small_strata)
                with torch.no_grad():
                    small_teacher=self.video(small_parts,teacher=True)
                    small_action=self.predict(small_parts,small_teacher['cache'])
                return self.measure_gt_mask_gradients(small_parts,update,small_rows,small_teacher,small_action)
            # Finish and release all small diagnostic graphs before constructing
            # the ordinary four-window teacher/student training graphs.
            with diagnostic_rng():
                measured=measure_in_chunks(rows,positions,calibration_chunk,measure_small)
        strata=[seed_for(self.config['seed'],'stratum',p)%4 for p in positions]
        parts=prepare(self.reference,self.reader,rows,positions,self.config['seed'],strata)
        with torch.no_grad():
            teacher=self.video(parts,teacher=True)
            teacher_action=self.predict(parts,teacher['cache'])
        self.measured_mask_gradients=(measured if split_calibration else self.measure_gt_mask_gradients(parts,update,rows,teacher,teacher_action) if measure_gt else None)
        student=self.video(parts,gates=self.gates(update,mode))
        # Frozen Action2B remains differentiable with respect to student K/V.
        action=self.predict(parts,student['cache'])
        ramp=1. if self.masks is None else schedule(update)['teacher']
        losses=objectives(action,teacher_action,parts[5],parts[0]['action_loss_is_pad'],
            self.reference.train_action_scheduler.training_weight(parts[4]),student['video'],teacher['video'],
            cache_terms(student['cache'],teacher['cache']),future_mask(rows,self.device),ramp)
        return losses

    def check_frozen(self):
        if any(p.requires_grad or p.grad is not None for p in self.reference.parameters()):
            raise RuntimeError('Frozen policy acquired gradients or trainable parameters')
