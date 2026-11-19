"""Prepare a new run from its configured demonstrations and model assets."""
import argparse
import json
from pathlib import Path
import torch
from bagel_robotics.data_selection.manifest import file_hash,immutable
from bagel_robotics._internal.adaptive_mask.geometry import Geometry,ParameterInventory
from bagel_robotics.adaptive_pruning.masks import Masks,SoftCost
from bagel_robotics.adaptive_pruning.config import load,identity
from bagel_robotics.adaptive_pruning.data import validate_panel,stream


def prepare(config):
    root=Path(config['root']);root.mkdir(parents=True,exist_ok=True)
    for name in ('checks','logs','mask/checkpoints','mask/diagnostics','recovery/checkpoints','recovery/diagnostics','exports'):
        (root/name).mkdir(parents=True,exist_ok=True)
    panel=json.loads(Path(config['panel_file']).read_text());fit,monitor=validate_panel(panel)
    normalizer=Path(config['demonstrations'])/'dataset_stats.json'
    if file_hash(normalizer)!=file_hash(Path(config['teacher_bundle'])/'dataset_stats.json'):
        raise ValueError('Demonstrations and base model need identical normalization')
    if config.get('normalization_sha256') and file_hash(normalizer)!=config['normalization_sha256']:
        raise ValueError('Configured normalization checksum mismatch')
    weights=torch.load(Path(config['teacher_bundle'])/'video.pt',map_location='cpu',mmap=True,weights_only=True)['dit']
    geometry=Geometry(**config['geometry']);inventory=ParameterInventory({k:v.shape for k,v in weights.items()},geometry)
    if inventory.original!=config['original_video_parameters']:raise ValueError('Base video tensor inventory changed')
    compact=None;selection=None
    if config['stage']=='mask':
        ranking=torch.load(config['initial_masks'],map_location='cpu',weights_only=False)['state']['controller']
        controller=Masks(geometry,ranking,config['initial_preferred'])
        selection,_=controller.selection();cost=SoftCost(inventory)
        if int(round(float(cost.parameters(controller.gates(0,'binary')))))!=inventory.count(selection):
            raise ValueError('Soft/binary physical cost mismatch')
    else:
        from bagel_robotics.adaptive_pruning.export import read_export_manifest
        compact=read_export_manifest(config['compact_model'],require_verified=True)
        if compact['geometry']!=config['geometry']:raise ValueError('Compact/base geometry mismatch')
        compact_root=Path(config['compact_model']) if compact.get('native_bundle') else Path(config['compact_model'])/'native'
        if file_hash(compact_root/'dataset_stats.json')!=file_hash(normalizer):
            raise ValueError('Compact model normalization differs from demonstrations/base model')
        base_action=torch.load(Path(config['teacher_bundle'])/'action.pt',map_location='cpu',mmap=True,weights_only=True)
        compact_action=torch.load(compact_root/'action.pt',map_location='cpu',mmap=True,weights_only=True)
        for field in ('action_expert','proprio_encoder'):
            if set(base_action[field])!=set(compact_action[field]) or any(not torch.equal(base_action[field][key],compact_action[field][key]) for key in base_action[field]):
                raise ValueError('Compact model must use the configured frozen action expert')
        del base_action,compact_action
        selection=compact['selection']
    samples=stream(panel,config['seed'],config['stage']=='recovery',
                   curated_recovery=config['stage']=='recovery' and config['recovery'].get('sample_mode','selected_windows')=='selected_windows')
    result=dict(identity=identity(config),panel_sha256=file_hash(config['panel_file']),
                normalization_sha256=file_hash(normalizer),dataset_manifest_sha256=file_hash(Path(config['demonstrations'])/'manifest.json'),shapes={k:list(v) for k,v in inventory.shapes.items()},
                original_parameters=inventory.original,initial_selection=selection,
                initial_binary_parameters=inventory.count(selection),
                counts=dict(fitting_windows=len(fit),monitor_windows=len(monitor),tasks=len(panel['tasks'])),
                sampled_windows_by_task=samples.sizes,
                assets={name:file_hash(Path(config['teacher_bundle'])/name) for name in ('video.pt','action.pt','dataset_stats.json')},
                encoder_files={str(path.relative_to(Path(config['wan_root']))):file_hash(path) for path in sorted(Path(config['wan_root']).rglob('*')) if path.is_file() and not any(part.startswith('.') for part in path.relative_to(Path(config['wan_root'])).parts)})
    immutable(root/'inputs.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    print(json.dumps(prepare(load(args.config)),indent=2))

if __name__=='__main__':main()
