"""Create content-pinned inputs and launch artifacts, never submit jobs."""
import argparse
import json
from pathlib import Path
import torch
from vpp2.full_mask.panel import file_hash, immutable, digest
from vpp2.adaptive_mask.geometry import Geometry, ParameterInventory
from .config import load, identity
from .controller import Masks, SoftCost
from .data import validate_panel, stream


def prepare(config):
    root=Path(config['root']);root.mkdir(parents=True,exist_ok=True)
    for name in ('checks','logs','slurm','mask/checkpoints','mask/diagnostics','recovery/checkpoints','recovery/diagnostics','exports','control'):
        (root/name).mkdir(parents=True,exist_ok=True)
    for name,entry in config['artifacts'].items():
        if file_hash(entry['path'])!=entry['sha256']:
            raise ValueError('Pinned artifact changed: '+name)
    if file_hash(config['panel_file'])!=config['panel_sha256']:
        raise ValueError('Panel hash changed')
    panel=json.loads(Path(config['panel_file']).read_text());fit,monitor=validate_panel(panel)
    for name in ('visual_review','data_cpu_receipt'):
        receipt=json.loads(Path(config['artifacts'][name]['path']).read_text())
        if receipt.get('passed') is not True or receipt.get('panel_identity')!=panel['identity']:
            raise ValueError('Data verification receipt missing/mismatched: '+name)
    from vpp2.adaptive_mask.audit import validate_manifest
    validate_manifest(json.loads(Path(config['artifacts']['audit_manifest']['path']).read_text()),panel)
    if digest({k:v for k,v in panel.items() if k!='identity'})!=panel['identity']:
        raise ValueError('Panel content identity changed')
    # The visual receipt is coarse screening, not proof every critical event is labelled.
    for base in (config['demonstrations'],config['teacher_bundle']):
        if file_hash(Path(base)/'dataset_stats.json')!=config['normalization_sha256']:
            raise ValueError('Original normalization changed')
    mask_path=Path(config['initial_masks'])
    if file_hash(mask_path)!=config['initial_masks_sha256']:
        raise ValueError('Mask initialization changed')
    # Trusted local experiment artifact contains Python/NumPy RNG state.
    old=torch.load(mask_path,map_location='cpu',weights_only=True)
    ranking=old['state']['controller']
    g=Geometry(**config['geometry'])
    controller=Masks(g,ranking,config['initial_preferred'])
    source=Path(config['teacher_bundle'])/'video.pt'
    weights=torch.load(source,map_location='cpu',mmap=True,weights_only=True)['dit']
    inv=ParameterInventory({k:v.shape for k,v in weights.items()},g)
    if inv.original!=config['original_video_parameters']:raise ValueError('Original inventory changed')
    soft=SoftCost(inv);selection,added=controller.selection()
    actual=inv.count(selection)
    if int(round(float(soft.parameters(controller.gates(0,'binary')))))!=actual:
        raise ValueError('Soft/binary physical cost mismatch')
    recovery=stream(panel,config['seed'],True)
    inputs=dict(identity=identity(config),panel_identity=panel['identity'],
        teacher={name:dict(path=str(Path(config['teacher_bundle'])/name),sha256=file_hash(Path(config['teacher_bundle'])/name))
                 for name in ('video.pt','action.pt','dataset_stats.json')},
        initial_mask_sha256=config['initial_masks_sha256'],
        counts=dict(fitting_windows=len(fit),monitor_windows=len(monitor),fitting_episodes=2906,
                    audit_episodes=280,audit_windows=1120,recovery_windows=sum(recovery.sizes.values())),
        initial_binary_parameters=actual,original_parameters=inv.original,
        shapes={k:list(v) for k,v in inv.shapes.items()},initial_selection=selection,
        recovery_windows_by_task=recovery.sizes,
        soft_cost_polynomial=dict(a=soft.a,b=soft.b,c=soft.c),
        audit_cross_experiment_independence_verified=False,
        visual_review_scope='Coarse screening only; no exhaustive semantic coverage claim',
        gpu_verified=False,jobs_submitted=False)
    immutable(root/'inputs.json',inputs)
    return inputs


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    result=prepare(load(a.config));print(json.dumps({k:v for k,v in result.items() if k not in ('shapes','initial_selection')},indent=2))

if __name__=='__main__':main()
