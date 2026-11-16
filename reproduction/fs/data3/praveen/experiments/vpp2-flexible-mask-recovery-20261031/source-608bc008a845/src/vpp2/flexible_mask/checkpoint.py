"""Atomic complete checkpoints, coupled trained-video/mask identity, exact restart."""
import json
from pathlib import Path
import shutil
import torch
import torch.distributed as dist
from .rng import rng_state,restore_rng
from vpp2.full_mask.panel import file_hash,digest
from vpp2.compact_kv.spec import atomic_json


def save(runtime,directory,state):
    root=Path(directory)/'checkpoints';tag=f"update_{state['update']:06d}"
    rngs=[None]*runtime.world;dist.all_gather_object(rngs,rng_state())
    client=dict(state,rank_rng=rngs,recovery_export_sha256=runtime.recovery_export_sha256,masks=None if runtime.masks is None else runtime.masks.state_dict(),
        mask_optimizer=None if runtime.mask_optimizer is None else runtime.mask_optimizer.state_dict(),
        cursor=state['update']*256)
    runtime.engine.save_checkpoint(str(root),tag=tag,client_state=client,save_latest=False)
    dist.barrier()
    own={str(p.relative_to(root/tag)):dict(size=p.stat().st_size,sha256=file_hash(p))
         for i,p in enumerate(sorted((root/tag).rglob('*.pt'))) if i%runtime.world==runtime.rank}
    parts=[None]*runtime.world;dist.all_gather_object(parts,own)
    if runtime.rank==0:
        files={k:v for part in parts for k,v in part.items()}
        if not files:raise RuntimeError('Empty checkpoint')
        receipt=dict(identity=state['identity'],update=state['update'],stage=state['stage'],files=files,
            recovery_export_sha256=runtime.recovery_export_sha256,
            mask_identity=None if runtime.masks is None else digest(runtime.masks.selection()[0]))
        receipt['checkpoint_identity']=digest(receipt)
        atomic_json(root/tag/'COMPLETE.json',receipt)
        tmp=root/'latest.partial';tmp.write_text(tag+'\n');tmp.replace(root/'latest')
        torch.save(dict(update=state['update'],checkpoint_identity=receipt['checkpoint_identity'],
                        scores=None if runtime.masks is None else {k:v.cpu() for k,v in runtime.masks.state_dict().items()},
                        selection=None if runtime.masks is None else runtime.masks.selection()[0]),
                   Path(directory)/f"masks_{state['update']:06d}.pt")
    dist.barrier()
    return root/tag


def load(runtime,directory,identity,stage,tag=None):
    root=Path(directory)/'checkpoints'
    if tag is None:
        if not (root/'latest').exists():return None
        tag=(root/'latest').read_text().strip()
    receipt=json.loads((root/tag/'COMPLETE.json').read_text())
    if receipt['identity']!=identity or receipt['stage']!=stage:raise ValueError('Checkpoint provenance mismatch')
    if receipt.get('recovery_export_sha256')!=runtime.recovery_export_sha256:
        raise ValueError('Recovery initialization export changed on resume')
    for i,(name,item) in enumerate(sorted(receipt['files'].items())):
        if i%runtime.world==runtime.rank:
            p=root/tag/name
            if p.stat().st_size!=item['size'] or file_hash(p)!=item['sha256']:
                raise ValueError('Corrupt checkpoint partition')
    dist.barrier()
    path,client=runtime.engine.load_checkpoint(str(root),tag=tag,load_lr_scheduler_states=False)
    if not path or client['identity']!=identity or client['cursor']!=client['update']*256:
        raise ValueError('Invalid checkpoint cursor/identity')
    if runtime.masks is not None:
        runtime.masks.load_state_dict(client['masks'],strict=True)
        runtime.mask_optimizer.load_state_dict(client['mask_optimizer'])
        if digest(runtime.masks.selection()[0])!=receipt['mask_identity']:
            raise ValueError('Mask and video checkpoint do not match')
    if runtime.engine.global_steps!=client['update']:raise ValueError('Optimizer update counter mismatch')
    restore_rng(client['rank_rng'][runtime.rank])
    return client


def retain(directory,best_tags):
    """Only this new stage's complete checkpoints: latest two plus best two."""
    root=Path(directory)/'checkpoints'
    complete=sorted(p for p in root.glob('update_*') if (p/'COMPLETE.json').exists())
    keep={p.name for p in complete[-2:]}|set(best_tags[:2])
    for p in complete:
        if p.name not in keep:shutil.rmtree(p)
