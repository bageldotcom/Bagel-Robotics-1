"""Pinned continuation from the completed no-K/V update-250 checkpoint."""
import argparse
import copy
import json
import math
from pathlib import Path
from statistics import mean

from vpp2.full_mask.panel import digest, file_hash, immutable
from vpp2.compact_kv.spec import atomic_json

SCHEMA = 'no-kv-recovery250-to1000-v1'
ALLOWED_SOURCE_CHANGES = {
    'src/vpp2/flexible_mask/config.py',
    'src/vpp2/flexible_mask/low_lr.py',
    'src/vpp2/flexible_mask/train.py',
    'src/vpp2/flexible_mask/recovery_extension.py',
    'src/vpp2/flexible_mask/recovery_extension_entry.py',
    'src/vpp2/flexible_mask/test_recovery_resume_gpu.py',
    'tests/flexible_mask/test_recovery_extension.py',
}
STORAGE_KEYS = ('artifact_storage', 'checkpoint_storage', 'load_scratch')


def validate(config):
    p = config.get('recovery_extension')
    if p is None:
        return
    fixed = dict(schema=SCHEMA, parent_update=250, parent_tag='update_000250',
                 target_update=1000, gradient_updates=[500, 1000],
                 condition='gt_better_than_historical250',
                 selection_metric='task_macro_ground_truth_native_action_mse')
    if set(p) != set(fixed) | {'parent_root', 'parent_identity', 'parent_pins'}:
        raise ValueError('Incomplete extension configuration')
    for name, value in fixed.items():
        if p.get(name) != value:
            raise ValueError('Extension setting changed: '+name)
    if not config.get('recovery_lr_branch') or not config.get('recovery_subset'):
        raise ValueError('Extension requires the existing no-K/V curated recovery')
    if not Path(p['parent_root']).is_absolute() or Path(p['parent_root']).resolve() == Path(config['root']).resolve():
        raise ValueError('Extension requires a separate root and an absolute parent path')
    if set(p['parent_pins']) != {'config.json', 'source_manifest.json', 'inputs.json',
                                'checks/cpu.json', 'checks/gradient_noninterference_gpu.json'}:
        raise ValueError('Missing parent evidence pins')
    if not 250 < p['target_update'] <= config['recovery_lr_branch']['schedule_endpoint']:
        raise ValueError('Extension exceeds the unchanged LR schedule')


def canonical_recipe(config):
    """Only output locations, run labels and the explicit extension may change."""
    c = copy.deepcopy(config)
    c.pop('recovery_extension', None)
    c.pop('root')
    c['wandb'].pop('name', None)
    c['wandb'].pop('group', None)
    for key in STORAGE_KEYS:
        c['recovery_lr_branch'].pop(key)
    return c


def verify_prepared(config):
    from .config import identity, sources
    validate(config)
    p = config['recovery_extension']; parent = Path(p['parent_root'])
    for name, expected in p['parent_pins'].items():
        if file_hash(parent/name) != expected:
            raise ValueError('Parent evidence changed: '+name)
    old = json.loads((parent/'config.json').read_text())
    if canonical_recipe(config) != canonical_recipe(old):
        raise ValueError('Extension changed the training recipe')
    previous = json.loads((parent/'source_manifest.json').read_text())
    if previous['identity'] != p['parent_identity'] or digest(dict(config=old, sources=previous['files'])) != p['parent_identity']:
        raise ValueError('Parent source/config identity mismatch')
    current = sources()
    changed = {name for name in set(previous['files']) | set(current)
               if previous['files'].get(name) != current.get(name)}
    if not changed or not changed <= ALLOWED_SOURCE_CHANGES:
        raise ValueError('Unapproved training-code changes: '+str(sorted(changed)))
    recorded = json.loads((Path(config['root'])/'source_manifest.json').read_text())
    if recorded != dict(identity=identity(config), files=current):
        raise ValueError('Extension source changed after verification')
    before = json.loads((parent/'inputs.json').read_text())
    after = json.loads((Path(config['root'])/'inputs.json').read_text())
    if before.pop('identity') != p['parent_identity'] or after.pop('identity') != identity(config) or before != after:
        raise ValueError('Data/model inputs changed in continuation')
    for name in ('cpu', 'gradient_noninterference_gpu'):
        evidence = json.loads((parent/f'checks/{name}.json').read_text())
        if evidence.get('passed') is not True or evidence['identity'] != p['parent_identity']:
            raise ValueError('Parent verification is not valid')
    return old, sorted(changed)


def window_ids(result):
    return sorted((r['window_id'], r['episode'], r['frame'], r['executed_steps']) for r in result['rows'])


def gt_error(result):
    return mean(v['student_error'] for k, v in result['actions'].items() if k.startswith('task/'))


def assess_validation(current, historical):
    from .selection import aggregate
    for result in (current, historical):
        rows = result['rows']; counts = {}
        for row in rows:
            counts[row['task']] = counts.get(row['task'], 0)+1
            for key in ('student_error', 'teacher_error'):
                if not math.isfinite(row[key]) or row[key] < 0:
                    raise ValueError('Invalid validation action error')
        if (result['update'] != 250 or len(rows) != 140 or len(counts) != 35 or set(counts.values()) != {4}
                or not result['native_10_step_sampler'] or result['simulation'] or result['full']
                or aggregate(rows) != result['actions']):
            raise ValueError('Incomplete or changed update-250 validation')
    if window_ids(current) != window_ids(historical):
        raise ValueError('Cannot compare different validation windows')
    new_error, old_error = gt_error(current), gt_error(historical)
    if not new_error < old_error:
        raise RuntimeError(f'Conditional extension withheld: update250 GT MSE {new_error} does not beat previous250 {old_error}')
    return dict(current250_gt_action_mse=new_error, historical250_gt_action_mse=old_error,
                relative_improvement=1-new_error/old_error,
                comparison='Same 140 offline native-action windows; not simulation evidence')


def bind_parent(config, create=False):
    """Bind the future checkpoint only after successful, complete pilot output."""
    from .config import identity
    old, changed = verify_prepared(config)
    p = config['recovery_extension']; parent = Path(p['parent_root'])
    ready = json.loads((parent/'control/review_ready.json').read_text())
    if ready.get('update') != 250 or any(ready.get(k) is not True for k in ('checkpoint_saved', 'native_export_saved', 'training_stopped_for_review')):
        raise ValueError('Pilot checkpoint/export is not complete')
    review = json.loads((parent/'control/review_250.json').read_text())
    if review['checkpoint250'] != str(parent/'recovery/checkpoints/update_000250'):
        raise ValueError('Wrong checkpoint in completed pilot review')
    checkpoint = parent/'recovery/checkpoints/update_000250'
    receipt = json.loads((checkpoint/'COMPLETE.json').read_text())
    if (receipt['identity'] != p['parent_identity'] or receipt['stage'] != 'recovery' or receipt['update'] != 250
            or receipt['checkpoint_identity'] != digest({k:v for k,v in receipt.items() if k != 'checkpoint_identity'})
            or receipt['recovery_export_sha256'] != config['recovery_subset']['export_manifest_sha256']):
        raise ValueError('Wrong update-250 training checkpoint')
    if sum(name.endswith('optim_states.pt') for name in receipt['files']) != config['train']['world_size']:
        raise ValueError('Missing optimizer shards in update-250 checkpoint')
    for name, entry in receipt['files'].items():
        path = checkpoint/name
        if not path.is_file() or path.stat().st_size != entry['size']:
            raise ValueError('Incomplete update-250 shard: '+name)
    sidecar = json.loads((checkpoint/'BEST_MODELS.json').read_text())
    if sidecar['checkpoint_identity'] != receipt['checkpoint_identity']:
        raise ValueError('Best-checkpoint references do not match restart state')
    current_path = parent/'recovery/diagnostics/native_000250.json'
    historical_path = Path(config['recovery_lr_branch']['comparison_root'])/'recovery/diagnostics/native_000250.json'
    current = json.loads(current_path.read_text()); historical = json.loads(historical_path.read_text())
    decision = assess_validation(current, historical)
    best = json.loads((parent/'recovery/best_models.json').read_text())
    from .recovery_best import verify_artifact
    if best['observed_through_update'] != 250 or len(best['best']) != 2:
        raise ValueError('Pilot best-two tracking is incomplete')
    for item in best['best']:
        verify_artifact(item)
        if item['artifact']['path'] not in sidecar['paths']:
            raise ValueError('Best policy is not protected by the parent restart')
    names = ['recovery/checkpoints/update_000250/COMPLETE.json',
             'recovery/checkpoints/update_000250/BEST_MODELS.json',
             'recovery/diagnostics/native_000250.json', 'recovery/diagnostics/gradients_000250.json',
             'recovery/best_models.json', 'control/review_ready.json', 'control/review_250.json']
    binding = dict(identity=identity(config), passed=True, parent_identity=p['parent_identity'],
                   checkpoint_identity=receipt['checkpoint_identity'], parent_update=250,
                   target_update=p['target_update'], checkpoint_bytes=sum(x['size'] for x in receipt['files'].values()),
                   parent_files={name:file_hash(parent/name) for name in names},
                   approved_source_changes=changed, best_updates=[x['update'] for x in best['best']],
                   best_paths=[x['artifact']['path'] for x in best['best']],
                   optimizer_partition_hashes_checked_on_load=True, **decision)
    path = Path(config['root'])/'checks/extension_parent.json'
    if path.exists():
        if json.loads(path.read_text()) != binding:
            raise ValueError('Bound parent checkpoint or validation changed')
    elif create:
        immutable(path, binding)  # CPU handoff job is the only writer.
    else:
        raise ValueError('CPU handoff must bind the checkpoint before GPU launch')
    return binding


def state_fields(config):
    return dict(recovery_extension_identity=digest(config['recovery_extension']))


def check_state(config, state):
    if not config.get('recovery_extension'):
        return
    p = config['recovery_extension']
    if (state is None or state.get('recovery_extension_identity') != state_fields(config)['recovery_extension_identity']
            or not p['parent_update'] <= state['update'] <= p['target_update']):
        raise ValueError('Wrong extension resume state')
    from .recovery_subset import check_state as check_subset
    check_subset(config, state)


def load_parent(runtime, config):
    from . import checkpoint, low_lr, recovery_subset, recovery_best
    from .config import identity
    from .rng import diagnostic_rng
    from .preflight import fingerprint
    import torch.distributed as dist
    binding = bind_parent(config)
    p = config['recovery_extension']; parent = Path(p['parent_root'])
    old = json.loads((parent/'config.json').read_text())
    state = checkpoint.load(runtime, parent/'recovery', p['parent_identity'], 'recovery', tag=p['parent_tag'])
    low_lr.check_state(old, state); recovery_subset.check_state(old, state)
    if state['update'] != 250 or state['cursor'] != 250*256 or state['recovery_subset_cursor'] != (250-55)*256:
        raise ValueError('Optimizer or data cursor was reset at continuation')
    if runtime.engine.global_steps != 250:
        raise ValueError('DeepSpeed did not restore the optimizer update counter')
    for value in state['best']:
        recovery_best.verify_artifact(value)
    if [x['artifact']['path'] for x in state['best']] != binding['best_paths']:
        raise ValueError('Saved best-two records differ from bound pilot results')
    state = dict(state, identity=identity(config),
                 extension_parent_checkpoint_identity=binding['checkpoint_identity'],
                 **low_lr.state_fields(config), **state_fields(config))
    check_state(config, state)
    with diagnostic_rng():
        # Metadata inspection must not alter the restored training random stream.
        from .recovery_gradients import adam_state
        stats = dict(rank=runtime.rank, global_step=runtime.engine.global_steps,
                     optimizer_fingerprint=fingerprint(adam_state(runtime.engine)),
                     curated_cursor=state['recovery_subset_cursor'])
        atomic_json(Path(config['root'])/f'checks/restored_state_rank{runtime.rank}.json', stats)
    dist.barrier()
    return state


def verify_restored_validation(runtime, result):
    config = runtime.config
    path = Path(config['recovery_extension']['parent_root'])/'recovery/diagnostics/native_000250.json'
    old = json.loads(path.read_text())
    if result['update'] != 250 or window_ids(result) != window_ids(old):
        raise ValueError('Restored checkpoint250 uses a different validation panel')
    before = {row['window_id']:row for row in old['rows']}
    maximum = 0.
    for row in result['rows']:
        reference = before[row['window_id']]
        for key in ('student_error', 'teacher_error', 'student_teacher_error'):
            a, b = row[key], reference[key]
            if not math.isclose(a, b, rel_tol=.001, abs_tol=1e-8):
                raise ValueError('Restored checkpoint250 action predictions changed: '+key)
            maximum = max(maximum, abs(a-b)/max(abs(b), 1e-8))
        for key in ('student_components', 'teacher_components'):
            if any(not math.isclose(a,b,rel_tol=.001,abs_tol=1e-8)
                   for a,b in zip(row[key],reference[key],strict=True)):
                raise ValueError('Restored checkpoint250 action components changed')
    if runtime.rank == 0:
        atomic_json(Path(config['root'])/'checks/restore250.json', dict(
            passed=True, update=250, full_panel_windows=140, max_relative_error=maximum,
            restored_training_state=True, first_new_update=251,
            next_curated_position=(250-55)*256, loss_recipe_unchanged=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['prepared','bind'], default='prepared');args=parser.parse_args()
    from .config import load
    config=load(args.config)
    if args.mode=='bind':
        print(json.dumps(bind_parent(config, create=True), indent=2))
    else:
        _,changes=verify_prepared(config)
        print(json.dumps(dict(passed=True, approved_source_changes=changes)))


if __name__=='__main__':
    main()
