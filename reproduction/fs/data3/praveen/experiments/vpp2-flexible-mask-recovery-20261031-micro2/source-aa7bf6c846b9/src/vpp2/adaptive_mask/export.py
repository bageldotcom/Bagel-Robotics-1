"""Strict heterogeneous FFN/head export; all native K/V output coordinates retained."""
from pathlib import Path
import json
import torch
from torch import nn
from vpp2.models.wan21_14b.wan_video_dit import WanVideoDiT,I2VCrossAttention,precompute_freqs_cis_3d
from vpp2.full_mask.panel import file_hash,immutable
from .geometry import Geometry,slice_parameter

def build_compact(kwargs,selection,geometry,device='meta'):
    geometry.validate(selection);h=len(selection['hidden_indices']);args=dict(kwargs)
    args.update(dim=h,hidden_dim=h,ffn_dim=geometry.ffn,num_layers=geometry.layers,
                num_heads=geometry.heads,attn_head_dim=geometry.head_dim)
    with torch.device(device):
        model=WanVideoDiT(**args)
        for i,b in enumerate(model.blocks):
            n=len(selection['ffn_indices'][i]);c=len(selection['cross_head_indices'][i])
            b.ffn[0]=nn.Linear(h,n);b.ffn[2]=nn.Linear(n,h);b.ffn_dim=n
            old=b.cross_attn
            b.cross_attn=I2VCrossAttention(h,geometry.head_dim,c,old.norm_q.eps,True)
    model.freqs=precompute_freqs_cis_3d(geometry.head_dim)
    return model

def sliced_state(teacher,compact,selection,geometry):
    states={};audit={};shapes={k:tuple(v.shape) for k,v in compact.state_dict().items()}
    if set(teacher)!=set(shapes):raise ValueError('Native tensor inventory differs')
    for name,value in teacher.items():
        out,rule=slice_parameter(name,value,selection,geometry)
        if tuple(out.shape)!=shapes[name]:raise ValueError('Wrong sliced shape: '+name)
        states[name]=out.clone();audit[name]=dict(source_shape=list(value.shape),shape=list(out.shape),rule=rule)
    return states,audit

def publish(config,selection,geometry,kwargs,identity):
    """Called only after a validated stable selection; never updates the teacher."""
    root=Path(config['root'])/'export';root.mkdir(parents=True,exist_ok=True)
    source=Path(config['teacher_bundle'])/'video.pt'
    teacher=torch.load(source,map_location='cpu',mmap=True,weights_only=True)['dit']
    model=build_compact(kwargs,selection,geometry);shapes={k:tuple(v.shape) for k,v in model.state_dict().items()}
    if set(teacher)!=set(shapes):raise ValueError('Export inventory differs')
    files=[];audit={};count=0
    # Bounded host RAM; tensor blocks are immutable, individually checksummed.
    groups=['embedding']+[f'block_{i:02d}' for i in range(geometry.layers)]
    for group in groups:
        selected={k:v for k,v in teacher.items() if (f'block_{int(k.split(".")[1]):02d}' if k.startswith('blocks.') else 'embedding')==group}
        chunk={}
        for name,value in selected.items():
            result,rule=slice_parameter(name,value,selection,geometry)
            if tuple(result.shape)!=shapes[name]:raise ValueError('Export shape mismatch')
            chunk[name]=result;count+=result.numel();audit[name]=dict(shape=list(result.shape),rule=rule)
        path=root/f'{group}.pt';tmp=path.with_suffix('.partial');torch.save(chunk,tmp);tmp.replace(path)
        files.append(dict(path=path.name,sha256=file_hash(path),keys=sorted(chunk)))
    manifest=dict(schema='adaptive-vpp2-native-export-v1',identity=identity,selection=selection,
        geometry=geometry.__dict__,model_kwargs=kwargs,parameters=count,files=files,tensors=audit,
        auxiliary_modules_required=False,policy_retention_established=False)
    immutable(root/'manifest.json',manifest)
    return manifest

def load_export(path,device='cpu',dtype=torch.bfloat16):
    root=Path(path);manifest=json.loads((root/'manifest.json').read_text())
    if manifest['schema']!='adaptive-vpp2-native-export-v1':raise ValueError('Unknown export schema')
    geometry=Geometry(**manifest['geometry']);model=build_compact(manifest['model_kwargs'],manifest['selection'],geometry)
    seen=set()
    for entry in manifest['files']:
        p=root/entry['path']
        if file_hash(p)!=entry['sha256']:raise ValueError('Export checksum mismatch')
        chunk=torch.load(p,map_location='cpu',mmap=True,weights_only=True)
        if set(chunk)!=set(entry['keys']) or seen&set(chunk):raise ValueError('Repeated/missing shard keys')
        seen.update(chunk);model.load_state_dict(chunk,strict=False,assign=True)
    if seen!=set(model.state_dict()):raise ValueError('Incomplete export')
    if sum(p.numel() for p in model.parameters())!=manifest['parameters']:raise ValueError('Export parameter count mismatch')
    return model.to(device=device,dtype=dtype)
