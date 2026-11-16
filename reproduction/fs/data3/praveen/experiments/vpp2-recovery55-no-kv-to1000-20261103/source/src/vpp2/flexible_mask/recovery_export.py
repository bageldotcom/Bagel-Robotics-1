"""Export final recovered weights in the ordinary native policy format."""
import json,shutil
from pathlib import Path
import torch
import torch.distributed as dist
from vpp2.full_mask.panel import file_hash,immutable
from vpp2.compact_kv.spec import atomic_json
from .config import identity
from .export import read_export_manifest


def publish(runtime,initial_export,update):
    c=runtime.config;m=read_export_manifest(initial_export,require_verified=True)
    state=runtime.engine._zero3_consolidated_16bit_state_dict(exclude_frozen_parameters=False)
    path=Path(c['root'])/'recovery/exports'/f'update_{update:06d}'
    if runtime.rank==0:
        checkpoint=json.loads((Path(c['root'])/f'recovery/checkpoints/update_{update:06d}/COMPLETE.json').read_text())
        if path.exists():
            completed=json.loads((path/'COMPLETE.json').read_text())
            if completed['identity']!=identity(c) or completed.get('checkpoint_identity')!=checkpoint['checkpoint_identity']:
                raise ValueError('Refusing changed recovered policy')
            if any(file_hash(path/name)!=value for name,value in completed['files'].items()):
                raise ValueError('Recovered checkpoint bytes changed')
            dist.barrier()
            return
        temp=path.with_name('.'+path.name+'.partial');temp.mkdir(parents=True)
        if any(not k.startswith('video.') for k in state):raise ValueError('Auxiliary training tensors in export')
        video={k.removeprefix('video.'):v for k,v in state.items()}
        if set(video)!=set(m['tensors']) or any(list(v.shape)!=m['tensors'][k]['shape'] for k,v in video.items()):
            raise ValueError('Recovery changed compact architecture')
        torch.save(dict(dit=video,step=update,origin_step=100000),temp/'video.pt')
        reloaded=torch.load(temp/'video.pt',map_location='cpu',weights_only=True,mmap=True)['dit']
        if any(not torch.equal(video[k],reloaded[k]) for k in video):raise ValueError('Recovery serialization changed weights')
        action=torch.load(Path(c['teacher_bundle'])/'action.pt',map_location='cpu',weights_only=True,mmap=True)
        action['origin_step']=action['step'];action['step']=update;action['video_checkpoint']={'path':'video.pt'}
        torch.save(action,temp/'action.pt')
        for name in ('dataset_stats.json','compact_architecture.json'):
            shutil.copyfile(Path(initial_export)/'native'/name,temp/name)
        immutable(temp/'COMPLETE.json',dict(identity=identity(c),update=update,stage='recovery',
            video_parameters=m['parameters'],initial_selection_sha256=m['mask_identity'],
            checkpoint_identity=checkpoint['checkpoint_identity'],initial_export_sha256=runtime.recovery_export_sha256,
            inference_teacher_required=False,mask_modules_required=False,simulation_completed=False,
            recovery_subset=c.get('recovery_subset'),
            recovery_kv_decay=c.get('recovery_kv_decay'),
            files={p.name:file_hash(p) for p in temp.iterdir() if p.is_file()}))
        temp.replace(path)
        atomic_json(Path(c['root'])/'recovered_policy.json',dict(identity=identity(c),path=str(path),update=update))
    dist.barrier()


def publish_best(runtime, initial_export, candidate):
    """Atomically export a validation-selected policy without serializing Adam state."""
    from .rng import diagnostic_rng
    import uuid
    c=runtime.config;update=candidate['update']
    m=read_export_manifest(initial_export,require_verified=True)
    if m['parameters'] != candidate['parameters']:
        raise ValueError('Selection and exported policy parameter count differ')
    path=Path(c['root'])/'recovery/best_models'/f'update_{update:06d}'
    with diagnostic_rng():
        state=runtime.engine._zero3_consolidated_16bit_state_dict(exclude_frozen_parameters=False)
        if runtime.rank==0:
            if any(not k.startswith('video.') for k in state):
                raise ValueError('Auxiliary tensors in best policy export')
            video={k.removeprefix('video.'):v for k,v in state.items()}
            if set(video)!=set(m['tensors']) or any(list(v.shape)!=m['tensors'][k]['shape'] for k,v in video.items()):
                raise ValueError('Best policy changed compact architecture')
            if path.exists():
                receipt=json.loads((path/'COMPLETE.json').read_text())
                if (receipt['identity']!=identity(c) or receipt['validation_sha256']!=candidate['validation_sha256']
                        or receipt['update']!=update or any(file_hash(path/n)!=h for n,h in receipt['files'].items())):
                    raise ValueError('Existing best export provenance differs')
                reloaded=torch.load(path/'video.pt',map_location='cpu',weights_only=True,mmap=True)['dit']
                if set(reloaded)!=set(video) or any(not torch.equal(video[k],reloaded[k]) for k in video):
                    raise ValueError('Replayed best export differs from live model')
            else:
                temp=path.with_name('.'+path.name+'.partial-'+uuid.uuid4().hex)
                temp.mkdir(parents=True)
                torch.save(dict(dit=video,step=update,origin_step=100000),temp/'video.pt')
                reloaded=torch.load(temp/'video.pt',map_location='cpu',weights_only=True,mmap=True)['dit']
                if any(not torch.equal(video[k],reloaded[k]) for k in video):
                    raise ValueError('Best policy serialization changed weights')
                action=torch.load(Path(c['teacher_bundle'])/'action.pt',map_location='cpu',weights_only=True,mmap=True)
                action['origin_step']=action['step'];action['step']=update
                action['video_checkpoint']={'path':'video.pt'}
                torch.save(action,temp/'action.pt')
                for name in ('dataset_stats.json','compact_architecture.json'):
                    shutil.copyfile(Path(initial_export)/'native'/name,temp/name)
                shutil.copyfile(candidate['validation_path'],temp/'validation.json')
                immutable(temp/'COMPLETE.json',dict(identity=identity(c),update=update,stage='recovery',
                    weights_only=True,video_parameters=m['parameters'],
                    initial_selection_sha256=m['mask_identity'],
                    recovery_export_sha256=runtime.recovery_export_sha256,
                    validation_sha256=candidate['validation_sha256'],gt_action_mse=candidate['gt_action_mse'],
                    selection_metric='task_macro_ground_truth_native_action_mse',
                    inference_teacher_required=False,mask_modules_required=False,simulation_completed=False,
                    recovery_subset=c.get('recovery_subset'),recovery_kv_decay=c.get('recovery_kv_decay'),
                    files={p.name:file_hash(p) for p in temp.iterdir() if p.is_file()}))
                temp.replace(path)
        dist.barrier()
    return dict(kind='native_policy',path=str(path),receipt_sha256=file_hash(path/'COMPLETE.json'),
                initial_export_sha256=runtime.recovery_export_sha256)
