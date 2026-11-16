"""Actual-manifest calibration, monitoring and physical parameter-budget contracts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from .panel import immutable,file_hash
from .calibration import sampling_plan,validate_rows
from .native_monitor import select_windows
from .controller import Controller
from .search_quality import schedule_end
from .train import identity
from vpp2.width_kv.slicing import slice_tensor


def verify(config,panel):
    rows,counts=sampling_plan(panel)
    synthetic=[dict(window_id=r['id'],task=r['task'],episode=r['episode'],stratum=s,teacher_error=1.) for r in rows for s in range(4)]
    validate_rows(synthetic,rows,counts,panel)
    native=select_windows(panel,config)
    if any(r['role']!='monitor' for r in native) or len(native)!=len({r['episode'] for r in native}):raise ValueError('Native monitor is not episode-disjoint')
    selection=Controller(config).selection()
    source=torch.load(Path(config['teacher_bundle'])/'video.pt',map_location='cpu',weights_only=True,mmap=True)['dit']
    total=0;teacher=0
    for name,value in source.items():
        teacher+=value.numel()
        meta=torch.empty(value.shape,device='meta',dtype=value.dtype)
        narrow,_=slice_tensor(name,meta,selection['hidden_indices'],selection['ffn_indices'],5120)
        total+=narrow.numel()
    if total!=config['approved_video_parameters']:raise ValueError(f'Physical budget changed: {total} != {config["approved_video_parameters"]}')
    return dict(passed=True,experiment_identity=identity(config,panel),panel_identity=panel['identity'],
        calibration_draws=len(synthetic),calibration_group_counts=dict(Counter(counts.values())),
        synthetic_calibration_values_only=True,native_windows=len(native),native_episodes=len({r['episode'] for r in native}),
        teacher_video_parameters=teacher,compact_video_parameters=total,removed_video_parameters=teacher-total,
        mask_parameters=selection['mask_parameters'],mask_schedule_end=schedule_end(config),
        minimum_stable_history_end=schedule_end(config)+config['train']['stable_checks']*config['train']['monitor_every'],
        policy_retention_established=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    config=json.loads(Path(args.config).read_text());panel=json.loads(Path(config['panel_file']).read_text())
    report=verify(config,panel);immutable(Path(config['root'])/'checks/search_contracts.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
