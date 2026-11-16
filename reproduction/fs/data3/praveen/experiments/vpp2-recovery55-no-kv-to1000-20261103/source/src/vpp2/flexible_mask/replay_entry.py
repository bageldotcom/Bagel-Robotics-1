"""Use the original training path; only the stop/save boundary is update 250."""
import argparse
import json
import os
from pathlib import Path
import sys
import torch.distributed as dist


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    root=Path(args.config).parent
    sys.path.insert(0,str(root/'control_tools'))
    from replay_common import read, verify_pins, atomic
    plan=read(root/'replay_plan.json');verify_pins(plan)
    decision=read(root/'control/decision.json')
    handoff=read(root/'control/handoff_status.json')
    if decision.get('trigger_replay') is not True or 'preserved_checkpoint' not in handoff:
        raise RuntimeError('Conditional replay is not authorized by a completed validation/preservation decision')
    if Path(os.environ.get('VPP2_CHECKPOINT_LOAD_TMPDIR','')).resolve()!=Path(plan['load_scratch']).resolve():
        raise RuntimeError('Use the planned temporary checkpoint storage')
    from .config import load
    from .train import run
    from .recovery_export import publish
    c=load(args.config)
    runtime,status=run(c,stage='recovery',recovery_export=c['recovery_subset']['export_path'])
    if status.get('update')!=250 or status.get('status')!='replay_target_reached':
        raise RuntimeError('Replay interrupted before the target; preserved checkpoint and fidelity report need review')
    publish(runtime,c['recovery_subset']['export_path'],250)
    if runtime.rank==0:
        atomic(root/'control/replay_training_complete.json',dict(update=250,checkpoint_saved=True,
            native_export_saved=True,bitwise_weight_equality_established=False,
            fidelity_report=str(root/'checks/replay_fidelity.json')))
    dist.barrier()


if __name__=='__main__':main()
