"""Export recovered compact weights with the frozen action expert and normalizer."""
import json
import shutil
from pathlib import Path
import torch
import torch.distributed as dist
from bagel_robotics.data_selection.manifest import file_hash,immutable
from bagel_robotics._internal.compact_kv.spec import atomic_json
from bagel_robotics.adaptive_pruning.config import identity
from bagel_robotics.adaptive_pruning.export import read_export_manifest


def publish(runtime,initial_export,update):
    config=runtime.config;manifest=read_export_manifest(initial_export,require_verified=True)
    state=runtime.engine._zero3_consolidated_16bit_state_dict(exclude_frozen_parameters=False)
    destination=Path(config['root'])/'recovery/exports'/f'update_{update:06d}'
    if runtime.rank==0:
        if destination.exists():
            receipt=json.loads((destination/'COMPLETE.json').read_text())
            if receipt['identity']!=identity(config) or any(file_hash(destination/name)!=sha for name,sha in receipt['files'].items()):
                raise ValueError('Existing recovered policy differs from this run')
            atomic_json(Path(config['root'])/'recovered_policy.json',
                        dict(identity=identity(config),path=str(destination),update=update))
            dist.barrier()
            return destination
        import uuid
        temporary=destination.with_name('.'+destination.name+'.partial-'+uuid.uuid4().hex);temporary.mkdir(parents=True)
        if any(not key.startswith('video.') for key in state):raise ValueError('Unexpected auxiliary training weights')
        video={key.removeprefix('video.'):value for key,value in state.items()}
        if set(video)!=set(manifest['tensors']) or any(list(value.shape)!=manifest['tensors'][key]['shape'] for key,value in video.items()):
            raise ValueError('Recovery changed compact architecture')
        torch.save(dict(dit=video,step=update),temporary/'video.pt')
        reloaded=torch.load(temporary/'video.pt',map_location='cpu',weights_only=True,mmap=True)['dit']
        if any(not torch.equal(video[key],reloaded[key]) for key in video):raise ValueError('Recovery serialization changed weights')
        action=torch.load(Path(config['teacher_bundle'])/'action.pt',map_location='cpu',weights_only=True,mmap=True)
        action['origin_step']=action['step'];action['step']=update;action['video_checkpoint']={'path':'video.pt'}
        torch.save(action,temporary/'action.pt')
        source=Path(initial_export) if manifest.get('native_bundle') else Path(initial_export)/'native'
        for name in ('dataset_stats.json','compact_architecture.json'):shutil.copyfile(source/name,temporary/name)
        immutable(temporary/'COMPLETE.json',dict(identity=identity(config),update=update,stage='recovery',
            video_parameters=manifest['parameters'],initial_selection_sha256=manifest['mask_identity'],
            initial_export_sha256=runtime.recovery_export_sha256,inference_teacher_required=False,
            mask_modules_required=False,simulation_completed=False,
            files={path.name:file_hash(path) for path in temporary.iterdir() if path.is_file()}))
        temporary.replace(destination)
        atomic_json(Path(config['root'])/'recovered_policy.json',dict(identity=identity(config),path=str(destination),update=update))
    dist.barrier()
