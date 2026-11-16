"""Physical pruning consumes the selected TRAINED student, never the teacher."""
import argparse,json,os
from pathlib import Path
import torch
import torch.distributed as dist
from vpp2.full_mask.panel import file_hash,digest,immutable
from vpp2.adaptive_mask.geometry import Geometry,slice_parameter
from vpp2.adaptive_mask.export import build_compact
from vpp2.compact_kv.spec import atomic_json
from .config import load,identity


def read_export_manifest(path,require_verified=False):
    path=Path(path);m=json.loads((path/'manifest.json').read_text())
    if m.get('schema')!='adaptive-vpp2-native-export-v1' or m.get('source_kind')!='trained_student':
        raise ValueError('Not a trained-student compact export')
    for name,expected in m.get('native_files',{}).items():
        if file_hash(path/'native'/name)!=expected:raise ValueError('Native deployment checkpoint changed')
    if require_verified:
        r=json.loads((path/'VERIFIED.json').read_text())
        if not r.get('passed') or r['manifest_sha256']!=file_hash(path/'manifest.json'):
            raise ValueError('Export has not passed exact numerical verification')
    return m


def export_state(path):
    path=Path(path);m=read_export_manifest(path,require_verified=True);state={}
    for entry in m['files']:
        p=path/entry['path']
        if file_hash(p)!=entry['sha256']:raise ValueError('Export weight checksum changed')
        part=torch.load(p,map_location='cpu',mmap=True,weights_only=True)
        if set(state)&set(part) or set(part)!=set(entry['keys']):raise ValueError('Duplicate/missing export tensors')
        state.update(part)
    if set(state)!=set(m['tensors']):raise ValueError('Incomplete export')
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
    from .preflight import require_receipt
    from .runtime import Runtime,distributed
    from .checkpoint import load as restore
    require_receipt(c,'gpu');rank,_,_=distributed(c);root=Path(c['root'])
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
