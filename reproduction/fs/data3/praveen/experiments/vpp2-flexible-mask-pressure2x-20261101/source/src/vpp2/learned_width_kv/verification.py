"""Fail-closed real-weight checks and verified physical width surgery."""
import argparse
import copy
import gc
import json
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import default_collate
from vpp2.compact_kv.model import prepare_batch,native_cache,action_prediction,infer_from_cache,VideoBranch,load_reference
from vpp2.compact_kv.validation import make_panel
from vpp2.compact_kv.spec import atomic_json,file_sha
from vpp2.compact_kv.blocks import install_blocks,TerminalCacheBlock
from vpp2.compact_kv.gpu_check import tensor_hash
from vpp2.compact_kv.train import assert_tree_exact
from vpp2.width_kv.slicing import slice_state,slice_tensor
from vpp2.models.wan21_14b.wan_video_dit import DiTBlock,precompute_freqs_cis_3d
from .common import identity,verify_receipt,build_video
from .runtime import initialize,student_engine,frozen_reference,relative_rms
from .masking import MaskedBlock,convert_norm,apply_gate
from .checkpointing import export_bundle
from .lifecycle import preserve_rng


def device_tree(value):
    if isinstance(value,torch.Tensor):return value.to(f'cuda:{torch.cuda.current_device()}')
    if isinstance(value,dict):return {k:device_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [device_tree(v) for v in value]
    return value


def compact_branch(config,state,selection):
    video=build_video(config,True)
    compact,audit=slice_state(state,{k:v.shape for k,v in video.state_dict().items()},selection)
    video.load_state_dict(compact,strict=True,assign=True)
    video.freqs=precompute_freqs_cis_3d(128);video.use_gradient_checkpointing=False
    install_blocks(video,());video.blocks[-1].__class__=TerminalCacheBlock
    return VideoBranch(video).eval().requires_grad_(False),compact,audit


@torch.no_grad()
def check_real_fp32_block(config,selection):
    index=(0,19,38)[dist.get_rank()%3]
    state=torch.load(Path(config['teacher_bundle'])/'video.pt',map_location='cpu',weights_only=True,mmap=True)['dit']
    prefix=f'blocks.{index}.';blockstate={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
    with torch.device('meta'):
        wide=DiTBlock(5120,128,40,13824,cross_attn_type='i2v',cross_attn_norm=True)
        narrow=DiTBlock(2048,128,40,8192,cross_attn_type='i2v',cross_attn_norm=True)
    wide.load_state_dict({k:v.float().clone() for k,v in blockstate.items()},strict=True,assign=True)
    sliced={k:slice_tensor(prefix+k,v,selection['hidden_indices'],selection['ffn_indices'],5120)[0].float() for k,v in blockstate.items()}
    narrow.load_state_dict(sliced,strict=True,assign=True)
    wide.__class__=MaskedBlock;wide.terminal_cache_only=False
    for norm in (wide.norm1,wide.norm2,wide.norm3):convert_norm(norm)
    from vpp2.compact_kv.blocks import CacheFullBlock
    narrow.__class__=CacheFullBlock
    device=f'cuda:{torch.cuda.current_device()}';wide.to(device);narrow.to(device)
    h=torch.zeros(5120,device=device);h[selection['hidden_indices']]=1
    f=torch.zeros(13824,device=device);f[selection['ffn_indices'][index]]=1
    gen=torch.Generator(device=device).manual_seed(8100+index)
    x=apply_gate(torch.randn(1,8,5120,device=device,generator=gen),h)
    tm=apply_gate(torch.randn(1,6,5120,device=device,generator=gen)*.1,h)
    ctx=apply_gate(torch.randn(1,259,5120,device=device,generator=gen),h)
    freqs=torch.ones(8,1,64,dtype=torch.complex128,device=device)
    mask=torch.ones(8,8,dtype=torch.bool,device=device)
    left=wide(x,tm,freqs,ctx,None,mask,h,f)
    right=narrow(x[...,selection['hidden_indices']],tm[...,selection['hidden_indices']],freqs,ctx[...,selection['hidden_indices']],None,mask)
    errors=[relative_rms(left[0][...,selection['hidden_indices']],right[0]),relative_rms(left[1],right[1]),relative_rms(left[2],right[2])]
    if max(errors)>=1e-5:raise AssertionError(f'FP32 real block {index} surgery mismatch: {errors}')
    del wide,narrow,state,blockstate,sliced,left,right
    gc.collect();torch.cuda.empty_cache()
    return dict(layer=index,relative_rms=errors)


def integration(config):
    rank,world,_,_=initialize(config);root=Path(config['root']);ident=identity(config)
    verify_receipt(root/'checks/cpu.json',ident)
    selection=json.loads((root/'calibration/initial_selection.json').read_text())
    fp32=check_real_fp32_block(config,selection)
    engine,controller,_=student_engine(config,'A',1,selection)
    panel=make_panel(config);sample=panel.dataset._get(panel.items[rank*16][0]);sample['pilot_position']=torch.tensor(-(rank*16+1))
    sample=default_collate([sample])
    reference=frozen_reference(config,1,check_sample=sample)
    prepared=prepare_batch(reference,sample,config['seed'])
    legacy=prepare_batch(reference,sample,config['seed'],encode_video_target=True)
    assert_tree_exact(prepared,legacy)
    with_u=prepare_batch(reference,sample,config['seed'],return_noise_u=True)
    assert_tree_exact(prepared,with_u[:7])
    # Change raw training targets before input preparation, not just an unused
    # field in an already-prepared dictionary. Observation/history stays fixed.
    changed_sample=dict(sample,video=torch.zeros_like(sample['video']),action=sample['action']+17.)
    changed_prepared=prepare_batch(reference,changed_sample,config['seed'])
    for key in ('condition_latents','context','context_mask','clip_fea'):
        assert_tree_exact(prepared[0][key],changed_prepared[0][key])
    for index in (1,2,6): assert_tree_exact(prepared[index],changed_prepared[index])
    del changed_prepared,changed_sample,legacy,with_u
    inputs,noise,tv,noisy,ta,target,mask=prepared
    engine.eval();reference.mot.training=False
    action_sha=tensor_hash(reference.action_expert)
    with torch.no_grad():
        teacher=native_cache(reference,inputs,noise,tv,mask)
        opened=engine(inputs,noise,tv,mask[:1950,:1950],None,None)
        for a,b in zip(teacher,opened):
            for key in ('k','v'):torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        a=action_prediction(reference,inputs,teacher,noisy,ta,mask)
        b=action_prediction(reference,inputs,opened,noisy,ta,mask)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        changed=dict(inputs,action=inputs['action']+17.,input_latents=torch.randn_like(noise))
        other=engine(changed,noise,tv,mask[:1950,:1950],None,None)
        for a,b in zip(opened,other):
            for key in ('k','v'):torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        del teacher,opened,other
        binary=engine(inputs,noise,tv,mask[:1950,:1950],*controller.gates(1000,'binary'))
        binary_cpu=[{k:v.cpu() for k,v in item.items()} for item in binary]
        flow=action_prediction(reference,inputs,binary,noisy,ta,mask).cpu()
        del binary
    state=torch.load(Path(config['teacher_bundle'])/'video.pt',map_location='cpu',weights_only=True,mmap=True)['dit']
    narrow,_,_=compact_branch(config,state,selection)
    narrow.to(f'cuda:{torch.cuda.current_device()}',dtype=torch.bfloat16)
    with torch.no_grad():
        compact=narrow(inputs,noise,tv,mask[:1950,:1950])
        errors=[[relative_rms(x[k],y[k].cpu()) for k in ('k','v')] for x,y in zip(binary_cpu,compact)]
        predicted=action_prediction(reference,inputs,compact,noisy,ta,mask).cpu()
        action_error=relative_rms(flow,predicted)
    if max(v for layer in errors for v in layer)>=.01 or action_error>=.01:
        raise AssertionError(f'BF16 binary/compact mismatch: action={action_error}, KV max={max(max(v) for v in errors)}')
    def pointers(model):
        return {getattr(p,'ds_tensor',p).data_ptr() for p in model.parameters() if getattr(p,'ds_tensor',p).numel()}
    if pointers(engine.module)&pointers(reference):raise RuntimeError('Student shares writable storage with frozen reference')
    del narrow,compact,predicted,state,binary_cpu
    gc.collect();torch.cuda.empty_cache()
    from .gradient_probe import action_only_backward
    gradient_check=action_only_backward(engine,controller,reference,inputs,noise,tv,noisy,ta,target,mask)
    if tensor_hash(reference.action_expert)!=action_sha:raise RuntimeError('Action weights changed')
    rows=[None]*world;dist.all_gather_object(rows,dict(rank=rank,fp32=fp32,kv_relative_rms=errors,
        action_relative_rms=action_error,action_sha256=action_sha,action_only_gradient_check=gradient_check))
    if rank==0:atomic_json(root/'checks/integration.json',dict(passed=True,identity=ident,rows=rows,
        all_open_native_exact=True,teacher_sharding_exact=True,prepared_legacy_exact=True,exposing_u_changes_no_draws=True,causality=True,raw_target_causality=True,
        action_only_gradients_verified=True,no_parameter_aliases=True))
    dist.barrier();dist.destroy_process_group()


@torch.no_grad()
@preserve_rng()
def verify_native_export(config,update,reference,ident):
    """Use only the exported native bundle; compare paired ten-Euler-step output."""
    root=Path(config['root']);rank=dist.get_rank();folder=root/'runs/B/exports'/f'update_{update:06d}'
    receipt=verify_receipt(folder/'COMPLETE.json',ident)
    if rank==0:
        for name,digest in receipt['files'].items():
            if file_sha(folder/name)!=digest:raise RuntimeError('Native export checksum mismatch')
    dist.barrier()
    fixture=torch.load(root/'runs/B/validation'/f'fixture_{update:06d}_soft_rank{rank}.pt',map_location='cpu',weights_only=True)
    local_config=dict(config,teacher_bundle=str(folder))
    native=load_reference(local_config,device=f'cuda:{torch.cuda.current_device()}')
    native.mot.training=False
    values=device_tree(fixture);inputs=values['inputs'];mask=values['mask']
    cache=native_cache(native,inputs,values['noise'],values['t_v'],mask)
    prediction=action_prediction(native,inputs,cache,values['noisy_action'],values['t_a'],mask)
    chunk=infer_from_cache(native,inputs,cache,mask,[values['position']],config['seed'])
    torch.testing.assert_close(prediction.cpu(),fixture['expected_flow'],rtol=0,atol=0)
    torch.testing.assert_close(chunk.cpu(),fixture['expected_chunk'],rtol=0,atol=0)
    if tensor_hash(native.action_expert)!=tensor_hash(reference.action_expert):raise RuntimeError('Export changed Action2B')
    del native,cache,values,prediction,chunk;gc.collect();torch.cuda.empty_cache()
    dist.barrier()
    if rank==0:atomic_json(folder/'RELOAD_CHECK.json',dict(passed=True,identity=ident,update=update,
        flow_exact=True,ten_euler_steps_exact=True,original_teacher_loaded=False,masks_loaded=False,windows=8))


def surgery(config):
    rank,world,_,_=initialize(config);root=Path(config['root']);ident=identity(config)
    verify_receipt(root/'runs/A/COMPLETE.json',ident)
    folder=root/'runs/A/exports/update_001000';receipt=verify_receipt(folder/'COMPLETE.json',ident)
    if file_sha(folder/'wide.pt')!=receipt['wide_sha256'] or file_sha(folder/'selection.json')!=receipt['selection_sha256']:
        raise RuntimeError('Phase A final export changed')
    payload=torch.load(folder/'wide.pt',map_location='cpu',weights_only=False,mmap=True)
    selection=json.loads((folder/'selection.json').read_text())
    narrow,state,audit=compact_branch(config,payload['dit'],selection)
    if rank==0:
        atomic_json(root/'surgery/selection.json',selection);atomic_json(root/'surgery/transfer_audit.json',audit)
        export_bundle(state,config,root/'surgery/native',1000,selection,ident)
    dist.barrier()
    del payload,state;gc.collect()
    reference=frozen_reference(config,1)
    narrow.to(f'cuda:{torch.cuda.current_device()}',dtype=torch.bfloat16)
    fixture=torch.load(root/'runs/A/validation'/f'fixture_001000_binary_rank{rank}.pt',map_location='cpu',weights_only=True)
    # Keep expected K/V on CPU, transferring only prepared inputs.
    expected=fixture.pop('expected_cache');values=device_tree(fixture)
    with torch.no_grad():
        cache=narrow(values['inputs'],values['noise'],values['t_v'],values['mask'][:1950,:1950])
        errors=[[relative_rms(x[k],y[k].cpu()) for k in ('k','v')] for x,y in zip(expected,cache)]
        flow=action_prediction(reference,values['inputs'],cache,values['noisy_action'],values['t_a'],values['mask'])
        error=relative_rms(flow.cpu(),fixture['expected_flow'])
    if max(max(x) for x in errors)>=.01 or error>=.01:raise AssertionError('Trained binary mask failed physical-surgery parity')
    rows=[None]*world;dist.all_gather_object(rows,dict(rank=rank,kv_relative_rms=errors,action_relative_rms=error))
    if rank==0:atomic_json(root/'checks/surgery.json',dict(passed=True,identity=ident,rows=rows,
        video_sha256=file_sha(root/'surgery/native/video.pt'),selection_sha256=file_sha(root/'surgery/selection.json'),
        compact_parameters=5587562304,teacher_required_at_inference=False))
    dist.barrier();dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('stage',choices=['integration','surgery']);a=p.parse_args()
    config=json.loads(Path(a.config).read_text())
    (integration if a.stage=='integration' else surgery)(config)


if __name__=='__main__':main()
