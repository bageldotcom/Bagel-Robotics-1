"""Approved eight-GPU branch, blocked until checkpoint55 restoration and GPU checks pass."""
import argparse
import json
from pathlib import Path
import torch.distributed as dist


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    from .config import load,identity,sources
    from .preflight import require_receipt
    from .low_lr import verify_parent,check_storage
    from .train import run
    from .recovery_export import publish
    from vpp2.compact_kv.spec import atomic_json
    c=load(a.config);root=Path(c['root'])
    recorded=json.loads((root/'source_manifest.json').read_text())
    if recorded['identity']!=identity(c) or recorded['files']!=sources():
        raise RuntimeError('Prepared source changed after verification')
    require_receipt(c,'cpu');require_receipt(c,'gradient_noninterference_gpu')
    verify_parent(c);check_storage(c)
    runtime,status=run(c,stage='recovery',recovery_export=c['recovery_subset']['export_path'])
    if status['status']=='review_at_250':
        publish(runtime,c['recovery_subset']['export_path'],250)
        if runtime.rank==0:
            atomic_json(root/'control/review_ready.json',dict(update=250,checkpoint_saved=True,
                native_export_saved=True,original55_preserved=True,original_replay_preserved=True,training_stopped_for_review=True,
                schedule_endpoint=2399,automatic_further_training=False))
    elif status['status']!='checkpointed_interruption':
        raise RuntimeError('Unexpected lower-LR endpoint: '+str(status))
    dist.barrier()


if __name__=='__main__':main()
