"""GT-first policy selection at every native validation; restart saves are separate."""
import json
import math
from pathlib import Path
import shutil
import torch.distributed as dist
from vpp2.compact_kv.spec import atomic_json
from vpp2.full_mask.panel import digest, file_hash
from .selection import aggregate

SCHEMA = 'recovery-best-two-gt-first-v1'


def candidate(result, parameters, validation_path):
    if (not result.get('native_10_step_sampler') or result.get('simulation') is not False
            or result.get('full') is not False):
        raise ValueError('Rank only the fixed native-action monitoring panel')
    rows = result['rows']
    counts = {}
    for row in rows:
        counts[row['task']] = counts.get(row['task'], 0) + 1
    if len(rows) != 140 or len(counts) != 35 or set(counts.values()) != {4}:
        raise ValueError('Ranking requires all 35 tasks and four windows per task')
    actions = aggregate(rows)
    if actions != result['actions']:
        raise ValueError('Validation aggregates do not match their per-window evidence')
    tasks = [v['student_error'] for k, v in actions.items() if k.startswith('task/')]
    return dict(update=result['update'], parameters=parameters, full_native_actions=actions,
                gt_action_mse=sum(tasks)/len(tasks),
                validation_path=str(Path(validation_path).resolve()),
                validation_sha256=file_hash(validation_path),
                window_identity=digest(sorted((r['window_id'],r['episode'],r['frame'],r['executed_steps']) for r in rows)),
                teacher_action_agreement_available=bool(result.get('action_agreement')))


def key(value):
    score = value['gt_action_mse']
    if not math.isfinite(score) or score < 0:
        raise ValueError('Invalid GT selection metric')
    # Exact score ties prefer smaller models, then the earlier checkpoint.
    # Teacher GT error and K/V error never replace the primary GT metric.
    return score, value['parameters'], value['update']


def select(candidates):
    by_update = {}
    windows = set()
    for value in candidates:
        windows.add(value['window_identity'])
        if value['update'] in by_update:
            before = by_update[value['update']]
            if before['gt_action_mse'] != value['gt_action_mse'] or before['validation_sha256'] != value['validation_sha256']:
                raise ValueError('Conflicting evidence for one checkpoint update')
            if 'artifact' in before and 'artifact' not in value:
                continue
        by_update[value['update']] = value
    if len(windows) > 1:
        raise ValueError('Cannot rank checkpoints measured on different windows')
    return sorted(by_update.values(), key=key)[:2]


def verify_artifact(value):
    p = Path(value['validation_path'])
    if file_hash(p) != value['validation_sha256']:
        raise ValueError('Selection validation changed')
    checked = candidate(json.loads(p.read_text()), value['parameters'], p)
    if any(checked[k] != value[k] for k in checked):
        raise ValueError('Selection score does not match validation evidence')
    artifact = value['artifact']; root = Path(artifact['path'])
    receipt_path = root/'COMPLETE.json'
    if not root.is_absolute() or file_hash(receipt_path) != artifact['receipt_sha256']:
        raise ValueError('Best model receipt is absent or changed')
    receipt = json.loads(receipt_path.read_text())
    if receipt['update'] != value['update'] or receipt['recovery_export_sha256'] != artifact['initial_export_sha256']:
        raise ValueError('Best model step or architecture differs from its evidence')
    if artifact['kind'] == 'training_checkpoint':
        if receipt['stage'] != 'recovery' or receipt['checkpoint_identity'] != artifact['checkpoint_identity']:
            raise ValueError('Wrong historical training checkpoint')
        if receipt['checkpoint_identity'] != digest({k:v for k,v in receipt.items() if k != 'checkpoint_identity'}):
            raise ValueError('Invalid historical checkpoint receipt')
        for name, item in receipt['files'].items():
            if not (root/name).is_file() or (root/name).stat().st_size != item['size']:
                raise ValueError('Missing historical model partition: '+name)
    elif artifact['kind'] == 'native_policy':
        if (receipt['validation_sha256'] != value['validation_sha256'] or receipt.get('weights_only') is not True
                or receipt['video_parameters'] != value['parameters']):
            raise ValueError('Policy export was not bound to its validation')
        if any(not (root/name).is_file() for name in receipt['files']):
            raise ValueError('Incomplete native policy export')
    else:
        raise ValueError('Unsupported best-model artifact')


def write_manifest(config, best, update, latest=None):
    from .config import identity
    atomic_json(Path(config['root'])/'recovery/best_models.json', dict(
        schema=SCHEMA, identity=identity(config), observed_through_update=update,
        selection_metric='task_macro_ground_truth_native_action_mse',
        comparison='fixed 140 held-out monitoring windows; not simulation',
        teacher_action_agreement_available=any(x.get('teacher_action_agreement_available',False) for x in best),
        best=select(best), latest_validation=latest,
        restart_checkpoint_interval=250, validation_interval=config.get('recovery_lr_branch',{}).get('validation_every',250)))


def initialize(config, state, rank):
    best = select(state['best'])
    if any(x['update'] > state['update'] for x in best):
        raise ValueError('Future validation cannot enter an earlier resumed state')
    for value in best:
        verify_artifact(value)
    if rank == 0:
        write_manifest(config, best, state['update'])
    dist.barrier()
    return best


def prune_exports(config, best, through_update):
    """Prune only our own completed weight-only exports after manifest publication."""
    from .config import identity
    root = Path(config['root'])/'recovery/best_models'
    keep = {str(Path(x['artifact']['path']).resolve()) for x in best}
    for restart in (Path(config['root'])/'recovery/checkpoints').glob('update_*'):
        if not (restart/'COMPLETE.json').is_file():
            continue
        sidecar = restart/'BEST_MODELS.json'
        if not sidecar.exists():
            return  # incomplete reference bookkeeping must never authorize deletion
        refs = json.loads(sidecar.read_text())
        receipt = json.loads((restart/'COMPLETE.json').read_text())
        if refs['checkpoint_identity'] != receipt['checkpoint_identity']:
            raise ValueError('Restart/best-model reference mismatch')
        keep.update(str(Path(x).resolve()) for x in refs['paths'])
    for p in root.glob('update_*'):
        if p.is_symlink() or not (p/'COMPLETE.json').is_file():
            continue
        receipt = json.loads((p/'COMPLETE.json').read_text())
        if (receipt.get('identity') == identity(config) and receipt.get('weights_only') is True
                and receipt['update'] <= through_update and str(p.resolve()) not in keep):
            shutil.rmtree(p)


def observe(runtime, initial_export, result, best, validation_path):
    """Called by every rank after every scheduled validation, including update 250."""
    dist.barrier()  # rank 0 has committed the validation JSON before it is referenced
    value = candidate(result, runtime.physical_parameters, validation_path)
    proposed = select(best+[value])
    if value['update'] in {x['update'] for x in proposed} and value['update'] not in {x['update'] for x in best}:
        from .recovery_export import publish_best
        value['artifact'] = publish_best(runtime, initial_export, value)
        proposed = select(best+[value])
    if runtime.rank == 0:
        write_manifest(runtime.config, proposed, result['update'], latest=value)
        prune_exports(runtime.config, proposed, result['update'])
    dist.barrier()
    return proposed


def record_restart_references(config, update, best):
    path=Path(config['root'])/'recovery/checkpoints'/f'update_{update:06d}'
    receipt=json.loads((path/'COMPLETE.json').read_text())
    atomic_json(path/'BEST_MODELS.json',dict(checkpoint_identity=receipt['checkpoint_identity'],
        paths=[x['artifact']['path'] for x in best]))
