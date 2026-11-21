"""Physical pruning consumes the selected TRAINED student, never the teacher."""
import argparse,json,os
from pathlib import Path
import torch
import torch.distributed as dist
from bagel_robotics.data_selection.manifest import file_hash, digest, immutable
from bagel_robotics._internal.adaptive_mask.geometry import Geometry, slice_parameter
from bagel_robotics._internal.adaptive_mask.export import build_compact
from bagel_robotics._internal.compact_kv.spec import atomic_json
from bagel_robotics.adaptive_pruning.config import load, identity


def initialization_identity(path):
    path=Path(path)
    metadata=path/'manifest.json' if (path/'manifest.json').is_file() else path/'COMPLETE.json'
    return file_hash(metadata)


def read_export_manifest(path,require_verified=False):
    """Load a numerical-verified export or checksum-verified compact bundle."""
    path=Path(path)
    if (path/'manifest.json').is_file():
        manifest=json.loads((path/'manifest.json').read_text())
        if manifest.get('schema')!='adaptive-vpp2-native-export-v1' or manifest.get('source_kind')!='trained_student':
            raise ValueError('Not a trained compact-model export')
        for name,expected in manifest.get('native_files',{}).items():
            if file_hash(path/'native'/name)!=expected:raise ValueError('Compact deployment checkpoint changed')
        if require_verified:
            receipt=json.loads((path/'VERIFIED.json').read_text())
            if not receipt.get('passed') or receipt['manifest_sha256']!=file_hash(path/'manifest.json'):
                raise ValueError('Export requires numerical verification')
        return manifest
    architecture=json.loads((path/'compact_architecture.json').read_text())
    if architecture.get('schema')!='vpp2-compact-channels-v1':raise ValueError('Unsupported compact architecture')
    receipt=json.loads((path/'COMPLETE.json').read_text())
    for name,expected in receipt['files'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe compact file path')
        expected=expected['sha256'] if isinstance(expected,dict) else expected
        if file_hash(path/relative)!=expected:raise ValueError('Compact model checksum mismatch: '+name)
    state=torch.load(path/'video.pt',map_location='cpu',mmap=True,weights_only=True)['dit']
    count=sum(value.numel() for value in state.values())
    if count!=architecture['parameters']:raise ValueError('Compact parameter inventory mismatch')
    return dict(architecture,source_kind='trained_student',native_bundle=True,
                mask_identity=digest(architecture['selection']),
                tensors={name:dict(shape=list(value.shape)) for name,value in state.items()},
                verification_kind='file_integrity_and_architecture')


def export_state(path):
    path=Path(path);manifest=read_export_manifest(path,require_verified=True)
    if manifest.get('native_bundle'):
        return torch.load(path/'video.pt',map_location='cpu',mmap=True,weights_only=True)['dit']
    state={}
    for entry in manifest['files']:
        source=path/entry['path']
        if file_hash(source)!=entry['sha256']:raise ValueError('Compact tensor checksum mismatch')
        part=torch.load(source,map_location='cpu',mmap=True,weights_only=True)
        if set(state)&set(part) or set(part)!=set(entry['keys']):raise ValueError('Duplicate/missing compact tensors')
        state.update(part)
    if set(state)!=set(manifest['tensors']):raise ValueError('Incomplete compact tensor inventory')
    return state


def write_export(config,trained,selection,kwargs,checkpoint_receipt,update):
    root=Path(config['root'])/'exports';destination=root/f'update_{update:06d}'
    if destination.exists():
        m=read_export_manifest(destination)
        if m['checkpoint_identity']!=checkpoint_receipt['checkpoint_identity']:raise ValueError('Refusing changed export')
        return destination
    temp=root/f'.update_{update:06d}.partial'
    if temp.exists():raise ValueError('Incomplete prior export requires inspection: '+str(temp))
    temp.mkdir(parents=True)
    g=Geometry(**config['geometry']);g.validate(selection)
    compact=build_compact(kwargs,selection,g)
    shapes={n:tuple(v.shape) for n,v in compact.state_dict().items()}
    if set(trained)!=set(shapes):raise ValueError('Trained student inventory mismatch')
    files=[];audit={};count=0
    for group in ['embedding']+[f'block_{i:02d}' for i in range(40)]:
        chunk={}
        for name,tensor in trained.items():
            if (f'block_{int(name.split(".")[1]):02d}' if name.startswith('blocks.') else 'embedding')!=group:continue
            out,rule=slice_parameter(name,tensor,selection,g)
            if tuple(out.shape)!=shapes[name]:raise ValueError('Wrong compact shape: '+name)
            chunk[name]=out.clone();count+=out.numel();audit[name]=dict(shape=list(out.shape),rule=rule)
        path=temp/(group+'.pt');torch.save(chunk,path)
        files.append(dict(path=path.name,sha256=file_hash(path),keys=sorted(chunk)))
    # Verification-only wide weights are the exact selected trained checkpoint.
    torch.save(dict(dit=trained,update=update,checkpoint_identity=checkpoint_receipt['checkpoint_identity']),temp/'trained_wide.pt')
    native=temp/'native';native.mkdir()
    native_state={}
    for item in files:
        native_state.update(torch.load(temp/item['path'],map_location='cpu',mmap=True,weights_only=True))
    torch.save(dict(dit=native_state,step=update,origin_step=100000),native/'video.pt')
    action=torch.load(Path(config['teacher_bundle'])/'action.pt',map_location='cpu',mmap=True,weights_only=True)
    action['origin_step']=action['step'];action['step']=update;action['video_checkpoint']={'path':'video.pt'}
    torch.save(action,native/'action.pt')
    import shutil
    shutil.copyfile(Path(config['teacher_bundle'])/'dataset_stats.json',native/'dataset_stats.json')
    native_spec=dict(schema='vpp2-compact-channels-v1',model_kwargs=kwargs,geometry=g.__dict__,selection=selection,parameters=count)
    immutable(native/'compact_architecture.json',native_spec)
    native_files={p.name:file_hash(p) for p in native.iterdir() if p.is_file()}
    manifest=dict(schema='adaptive-vpp2-native-export-v1',identity=identity(config),source_kind='trained_student',
        checkpoint_identity=checkpoint_receipt['checkpoint_identity'],mask_identity=digest(selection),update=update,
        selection=selection,geometry=g.__dict__,model_kwargs=kwargs,parameters=count,files=files,tensors=audit,native_files=native_files,
        verification_source_sha256=file_hash(temp/'trained_wide.pt'),
        action_checkpoint='native/action.pt',
        auxiliary_modules_required=False,policy_retention_established=False)
    immutable(temp/'manifest.json',manifest);temp.replace(destination)
    return destination


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();c=load(a.config)
    from bagel_robotics.adaptive_pruning.runtime import Runtime, distributed
    from bagel_robotics._internal.training.checkpoint import load as restore
    rank,_,_=distributed(c);root=Path(c['root'])
    selected=json.loads((root/'selected.json').read_text())
    if selected['identity']!=identity(c) or selected['status']!='ready_for_export':raise ValueError('Missing qualified selection')
    candidate=selected['candidate'];runtime=Runtime(c)
    restore(runtime,root/'mask',identity(c),'mask',tag=candidate['checkpoint'])
    selection=runtime.masks.selection()[0]
    if selection!=candidate['selection']:raise ValueError('Mask/checkpoint pairing changed')
    receipt=json.loads((root/'mask/checkpoints'/candidate['checkpoint']/'COMPLETE.json').read_text())
    if receipt['checkpoint_identity']!=candidate['checkpoint_identity']:raise ValueError('Selected video weights changed')
    state=runtime.engine._zero3_consolidated_16bit_state_dict(exclude_frozen_parameters=False)
    if rank==0:
        if any(not k.startswith('video.') for k in state):raise ValueError('Unexpected auxiliary export weights')
        path=write_export(c,{k.removeprefix('video.'):v for k,v in state.items()},selection,runtime.kwargs,receipt,candidate['update'])
        atomic_json(root/'export_pending_verification.json',dict(path=str(path),identity=identity(c),verified=False))
    dist.barrier()

if __name__=='__main__':main()
