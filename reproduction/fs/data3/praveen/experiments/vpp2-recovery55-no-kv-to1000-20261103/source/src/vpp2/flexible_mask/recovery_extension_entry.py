"""Resume full update250 training state and stop at the authorized update1000."""
import argparse
import json
from pathlib import Path
import torch.distributed as dist


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    from .config import load
    from .preflight import require_receipt
    from .low_lr import verify_parent,check_storage
    from .recovery_extension import verify_prepared,bind_parent
    from .train import run
    from .recovery_export import publish
    from vpp2.compact_kv.spec import atomic_json
    config=load(args.config);root=Path(config['root'])
    verify_prepared(config)
    for name in ('cpu','gradient_noninterference_gpu','recovery_resume_gpu'):
        require_receipt(config,name)
    verify_parent(config);check_storage(config);bind_parent(config)
    runtime,status=run(config,stage='recovery',recovery_export=config['recovery_subset']['export_path'])
    target=config['recovery_extension']['target_update']
    if status['status']=='extension_complete' and status['update']==target:
        publish(runtime,config['recovery_subset']['export_path'],target)
        if runtime.rank==0:
            best=json.loads((root/'recovery/best_models.json').read_text())
            from .recovery_best import verify_artifact
            if len(best['best'])!=2:raise RuntimeError('Final best-two records are incomplete')
            for value in best['best']:verify_artifact(value)
            atomic_json(root/'control/completed.json',dict(update=target,checkpoint_saved=True,
                native_export_saved=True,best_two_preserved=True,
                best_updates=[v['update'] for v in best['best']],
                original250_preserved=True,automatic_further_training=False))
    elif status['status']!='checkpointed_interruption':
        raise RuntimeError('Unexpected continuation endpoint: '+str(status))
    dist.barrier()


if __name__=='__main__':main()
