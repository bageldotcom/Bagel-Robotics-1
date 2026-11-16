"""One frozen policy per rank; teacher and masked contexts share immutable weights."""
import hashlib
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from vpp2.compact_kv.model import load_reference,action_prediction,native_cache
from vpp2.compact_kv.losses import action_error,kv_terms
from vpp2.learned_width_kv.masking import install_masks
from vpp2.learned_width_kv.reliability import action_losses,kv_losses
from .data import WindowReader,collate,prepared
from .controller import Controller
from .panel import digest,file_hash,immutable


def distributed():
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    if world>1:dist.init_process_group('nccl',device_id=torch.device('cuda',local))
    return rank,world


def barrier():
    if dist.is_initialized():dist.barrier()


def gather(value):
    if not dist.is_initialized():return [value]
    outputs=[None]*dist.get_world_size();dist.all_gather_object(outputs,value);return outputs


def verify_checkpoint_receipt(config,receipt,rank):
    from vpp2.critical_windows.inference import model_identity
    value=[model_identity(config) if rank==0 else None]
    if dist.is_initialized():dist.broadcast_object_list(value,src=0)
    if value[0]!=receipt.get('checkpoint_hashes'):raise ValueError('Native checkpoint bytes changed since GPU preflight')


def model_versions(model):return {n:p._version for n,p in model.named_parameters()}


def check_frozen(model,versions):
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError('Frozen native policy acquired trainability/gradients')
    if model_versions(model)!=versions:raise RuntimeError('A frozen policy tensor was modified')


def initialize(config):
    model=load_reference(config)
    if any(isinstance(m,torch.nn.Dropout) and m.p>0 for m in model.modules()):
        raise RuntimeError('Checkpointing training flag would activate dropout')
    # Only enable native recomputation. Encoders stay in evaluation mode; no
    # teacher parameter is ever put in an optimizer.
    model.mot.training=True
    model.video_expert.use_gradient_checkpointing=True
    model.action_expert.use_gradient_checkpointing=True
    reader=WindowReader(config)
    model._full_mask_conditioning_cache=reader.runtime_cache
    return model,reader,Controller(config).cuda()


def install(model):
    branch=install_masks(model.video_expert);branch.training=True
    return branch


def cache(branch,parts,gates=(None,None)):
    inp,nv,tv,_,_,_,mask=parts
    return branch(inp,nv,tv,mask[:1950,:1950],*gates)


def predict(model,parts,c):return action_prediction(model,parts[0],c,parts[3],parts[4],parts[6])


def step_loss(model,branch,controller,parts,rows,strata,thresholds,update,config):
    with torch.no_grad():
        teacher_cache=cache(branch,parts);teacher=predict(model,parts,teacher_cache)
    gates=controller.gates(update);student_cache=cache(branch,parts,gates);student=predict(model,parts,student_cache)
    inp=parts[0];time_weight=model.train_action_scheduler.training_weight(parts[4]).float()
    gt,kd,raw,w,reasons=action_losses(student,teacher,parts[5],inp['action_loss_is_pad'],time_weight,
                                   [r['task'] for r in rows],strata,thresholds)
    raw_kv,applied_kv,layer=kv_losses(kv_terms(student_cache,teacher_cache),w)
    reg,penalties=controller.regularizer(update,*gates)
    cfg=config['train'];weight=cfg['kv_weight']*min((update+1)/cfg['kv_ramp_updates'],1.)
    loss=gt+cfg['teacher_action_weight']*kd+weight*applied_kv+reg
    metrics=torch.stack([gt.detach(),kd.detach(),raw.detach(),applied_kv.detach(),raw_kv.detach(),
                         reg.detach(),w.mean(),reasons['better_student'].float().mean(),
                         reasons['worse_than_zero'].float().mean(),*penalties.values(),*layer.flatten()])
    return loss,metrics


def calibrate_teacher(config,model,branch,reader,panel,rank,world):
    from .calibration import run
    return run(config,model,branch,reader,panel,rank,world)


def verify_gpu(config,panel,rank,world):
    from vpp2.compact_kv.gpu_check import tensor_hash
    from vpp2.critical_windows.inference import model_identity,deterministic_setup
    deterministic_setup(0)
    checkpoint_hashes=model_identity(config)
    model,reader,controller=initialize(config)
    row=next(r for r in panel['windows'] if r['role']=='selection_initial')
    # Always establish live encoder truth, even when restarting a preflight.
    saved_reader_cache=reader.runtime_cache;reader.runtime_cache=None
    parts=prepared(model,collate(reader,[row],[12345]),config['seed'],[2])
    reader.runtime_cache=saved_reader_cache
    versions=model_versions(model)
    # Full-module digests supplement tensor version checks for this real update.
    before_video=tensor_hash(model.video_expert);before_action=tensor_hash(model.action_expert)
    with torch.no_grad():
        original=native_cache(model,parts[0],parts[1],parts[2],parts[6]);original_action=predict(model,parts,original)
    branch=install(model)
    with torch.no_grad():
        opened=cache(branch,parts);open_action=predict(model,parts,opened)
        same=all(torch.equal(x[k],y[k]) for x,y in zip(original,opened) for k in ('k','v'))
        if not same or not torch.equal(original_action,open_action):raise AssertionError('All-open cache/action parity failed')
    del original,opened,original_action,open_action
    h,f=controller.gates(35)
    out=predict(model,parts,cache(branch,parts,(h,f)))
    loss=action_error(out,parts[5],parts[0]['action_loss_is_pad']).mean()
    loss.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in controller.parameters()):
        raise AssertionError('Missing/nonfinite gate gradients')
    if not (controller.ffn_scores.grad.abs().sum(-1)>0).all() or not controller.hidden_scores.grad.abs().sum()>0:
        raise AssertionError('Action gradients do not reach all action-visible mask groups')
    optimizer=torch.optim.AdamW(controller.parameters(),lr=.01,weight_decay=0.)
    optimizer.step();optimizer.zero_grad(set_to_none=True);check_frozen(model,versions)
    if tensor_hash(model.video_expert)!=before_video or tensor_hash(model.action_expert)!=before_action:
        raise AssertionError('Frozen native weights changed')
    # No direct future-target leakage into action conditioning.
    old_cache=reader.runtime_cache;reader.runtime_cache=None
    sample=collate(reader,[row],[12345]);reader.runtime_cache=old_cache
    # Deliberately change the unused inputs; comparing zeros with zeros is vacuous.
    original_future=sample['video'][:,:,9:].clone()
    sample['video'][:,:,9:]=1.0-original_future
    if torch.equal(original_future,sample['video'][:,:,9:]):raise AssertionError('Future perturbation was ineffective')
    changed=prepared(model,sample,config['seed'],[2])
    with torch.no_grad():
        a=predict(model,parts,cache(branch,parts));b=predict(model,changed,cache(branch,changed))
    if not torch.equal(a,b):raise AssertionError('Unused future RGB changes native policy output')
    if reader.runtime_cache is not None:
        loaded=collate(reader,[row],[12345])
        if '_native_prepared' not in loaded:raise AssertionError('Conditioning cache not populated')
        reused=prepared(model,loaded,config['seed'],[2])
        for i,(a,b) in enumerate(zip(parts,reused)):
            if isinstance(a,dict):
                for key in a:
                    if torch.is_tensor(a[key]):
                        if not torch.equal(a[key],b[key]):raise AssertionError(f'Cached input differs: {key}')
                    elif a[key]!=b[key]:raise AssertionError('Cached scalar differs')
            elif not torch.equal(a,b):raise AssertionError(f'Cached noise/target/mask differs: {i}')
        new_noise=prepared(model,collate(reader,[row],[12346]),config['seed'],[2])
        if torch.equal(parts[1],new_noise[1]) or torch.equal(parts[3],new_noise[3]):raise AssertionError('Conditioning cache froze training noise')
        del loaded,reused,new_noise
    # Validate the exact native sampler with raw observations before any training.
    from .native_monitor import select_windows,reader_manifest,inference_mode
    from vpp2.critical_windows.inference import NativeInferenceReader,masked_prefill
    entries=select_windows(panel,config);chosen=[next(r for r in entries if r['suite']==suite) for suite in ('memory','precision')]
    native_reader=NativeInferenceReader(config,reader_manifest(chosen))
    hard=controller.gates(0,'binary');native_checks=[]
    with inference_mode(model,branch):
        for r in chosen:
            inputs,_=native_reader.read(r['episode'],r['frame'])
            native=model.infer_action(**inputs)['action']
            repeat=model.infer_action(**inputs)['action']
            with masked_prefill(model,branch):opened=model.infer_action(**inputs)['action']
            if not torch.equal(native,repeat) or not torch.equal(native,opened):raise AssertionError('Native ten-step sampled action repeat/all-open parity failed')
            with masked_prefill(model,branch,*hard):binary=model.infer_action(**inputs)['action']
            if binary.shape!=(32,16) or not torch.isfinite(binary).all():raise AssertionError('Invalid binary native actions')
            native_checks.append(dict(task=r['task'],episode=r['episode'],frame=r['frame'],repeat_equal=True,all_open_equal=True,binary_finite=True))
    check_frozen(model,versions)
    from .train import identity
    record=dict(experiment_identity=identity(config,panel),checkpoint_hashes=checkpoint_hashes,native_sampler_checks=native_checks,conditioning_cache_exact=True,
                conditioning_cache_noise_not_frozen=True,passed=True,all_open_bitwise_equal=True,frozen_hashes_unchanged=True,
                action_gradients_reach_39_ffn_masks=True,future_target_invariance=True,future_rgb_was_perturbed=True,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                micro_batch=1,gpu=torch.cuda.get_device_name(),panel_identity=panel['identity'])
    immutable(Path(config.get('gpu_receipt_file',Path(config['root'])/'checks/gpu.json')),record)
    print(json.dumps(record),flush=True)
