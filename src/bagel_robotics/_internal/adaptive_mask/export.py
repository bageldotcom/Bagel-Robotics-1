"""Construct and reload the physically channel-compact video expert."""
import json
from pathlib import Path
import torch
from torch import nn
from bagel_robotics.models.backbone.wan21_14b.wan_video_dit import WanVideoDiT,I2VCrossAttention,precompute_freqs_cis_3d
from bagel_robotics.data_selection.manifest import file_hash
from bagel_robotics._internal.adaptive_mask.geometry import Geometry,slice_parameter

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
