"""Atomic verified phase checkpoints; two recent states plus phase boundaries."""
import json
import os
from pathlib import Path
import shutil
import torch
import torch.distributed as dist
from vpp2.compact_kv.checkpointing import rng_state
from vpp2.compact_kv.spec import atomic_json,file_sha
from .common import architecture
from .lifecycle import preserve_rng


@preserve_rng()
def save_training(engine,controller,mask_optimizer,directory,phase,update,config,ident,extra=None):
    rank,world=dist.get_rank(),dist.get_world_size()
    states=[None]*world;dist.all_gather_object(states,rng_state())
    global_update=update+(1000 if phase=='B' else 0)
    client=dict(schema='learned-width-training-state-v1',phase=phase,update=update,
        global_update=global_update,next_global_sample=global_update*config['global_batch'],identity=ident,
        thresholds_sha256=file_sha(Path(config['root'])/'calibration/thresholds.json'),
        rank_rng=states,controller=None if controller is None else controller.state_dict(),
        mask_optimizer=None if mask_optimizer is None else mask_optimizer.state_dict(),extra=extra or {})
    root=Path(directory)/'checkpoints';tag=f'update_{update:06d}'
    engine.save_checkpoint(str(root),tag=tag,client_state=client,save_latest=False)
    dist.barrier()
    if rank==0:
        files=sorted((root/tag).rglob('*.pt'))
        if not files or any(p.stat().st_size==0 for p in files): raise RuntimeError('Incomplete checkpoint partitions')
        atomic_json(root/tag/'COMPLETE.json',dict(passed=True,phase=phase,update=update,identity=ident,
            files={str(p.relative_to(root/tag)):dict(size=p.stat().st_size,sha256=file_sha(p)) for p in files}))
        latest=root/'latest.partial';latest.write_text(tag);latest.replace(root/'latest')
        verified=sorted(p for p in root.glob('update_*') if p.is_dir() and (p/'COMPLETE.json').exists())
        retain=set(verified[-2:])|{p for p in verified if p.name=='update_001000'}
        for old in verified:
            if old not in retain: shutil.rmtree(old)
    dist.barrier()


def export_bundle(state,config,destination,update,selection,ident):
    destination=Path(destination)
    if destination.exists():
        receipt=json.loads((destination/'COMPLETE.json').read_text())
        if receipt['identity']!=ident: raise RuntimeError('Refusing to overwrite different export')
        for name,digest in receipt['files'].items():
            if file_sha(destination/name)!=digest: raise RuntimeError('Existing export corrupted')
        return
    temp=destination.with_name(destination.name+'.partial')
    temp.mkdir(parents=True,exist_ok=True)
    arch=architecture(selection)
    torch.save(dict(dit=state,step=update,origin_step=100000,compact_architecture=arch),temp/'video.pt')
    source=Path(config['teacher_bundle'])
    action=torch.load(source/'action.pt',map_location='cpu',weights_only=True,mmap=True)
    from vpp2.checkpoint_compat import canonicalize_config
    action['resolved_config']=canonicalize_config(action['resolved_config'])
    action['resolved_config']['model']['video_dit_config'].update(dim=2048,hidden_dim=2048,ffn_dim=8192,num_layers=40,num_heads=40,attn_head_dim=128)
    action['origin_step'],action['step']=int(action['step']),update
    action['video_checkpoint']={'path':'video.pt'}
    torch.save(action,temp/'action.pt')
    shutil.copyfile(source/'dataset_stats.json',temp/'dataset_stats.json')
    atomic_json(temp/'compact_architecture.json',arch)
    atomic_json(temp/'COMPLETE.json',dict(passed=True,identity=ident,update=update,inference_teacher_required=False,
        files={p.name:file_sha(p) for p in temp.iterdir() if p.is_file() and p.name!='COMPLETE.json'}))
    temp.replace(destination)


@preserve_rng()
def export_phase(engine,controller,config,phase,update,ident):
    state=engine._zero3_consolidated_16bit_state_dict(exclude_frozen_parameters=False)
    root=Path(config['root'])
    if dist.get_rank()==0:
        video={k.removeprefix('video.'):v for k,v in state.items()}
        if any(not k.startswith('video.') for k in state): raise RuntimeError('Unexpected student parameters')
        folder=root/'runs'/phase/'exports'/f'update_{update:06d}'
        if phase=='A':
            folder.mkdir(parents=True,exist_ok=True)
            temp=folder/'wide.pt.partial'
            torch.save(dict(dit=video,controller=controller.state_dict(),update=update,identity=ident),temp)
            temp.replace(folder/'wide.pt')
            atomic_json(folder/'selection.json',controller.selection())
            atomic_json(folder/'COMPLETE.json',dict(passed=True,identity=ident,update=update,
                wide_sha256=file_sha(folder/'wide.pt'),selection_sha256=file_sha(folder/'selection.json')))
        else:
            selection=json.loads((root/'surgery/selection.json').read_text())
            export_bundle(video,config,folder,update,selection,ident)
    dist.barrier()
