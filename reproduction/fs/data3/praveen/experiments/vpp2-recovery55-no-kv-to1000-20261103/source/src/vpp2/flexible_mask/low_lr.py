"""Recovery from checkpoint 55 without K/V matching; replay250 stays external."""
import copy
import json
import math
from pathlib import Path
import shutil

SCHEMA = 'recovery55-low-lr-no-kv-v1'


def validate(config):
    p = config['recovery_lr_branch']
    fixed = dict(schema=SCHEMA, parent_update=55, parent_tag='update_000055',
        hold_through=250, review_update=250, schedule_endpoint=2399,
        validation_every=50, checkpoint_every=250,
        gradient_updates=[55, 150, 250], learning_rate=1e-6,
        final_lr_fraction=.1, diagnostic_windows_per_task=2, diagnostic_noise_draws=2)
    for key, value in fixed.items():
        if p.get(key) != value:
            raise ValueError('No-K/V recipe changed: '+key)
    for key in ('parent_root', 'comparison_root', 'checkpoint_storage', 'artifact_storage',
                'load_scratch', 'baseline55_validation'):
        if not Path(p[key]).is_absolute():
            raise ValueError('Absolute paths required: '+key)
    if Path(config['root']).resolve() in {Path(p[k]).resolve() for k in ('parent_root','comparison_root')}:
        raise ValueError('Parent and reconstructed checkpoint must remain isolated')
    if (p['parent_root'] != config['recovery_subset']['parent_root'] or
            p['parent_identity'] != config['recovery_subset']['parent_identity']):
        raise ValueError('Wrong checkpoint55 parent')
    if config['recovery'] != dict(updates=2344, learning_rate=1e-6, warmup_updates=0, final_lr_fraction=.1):
        raise ValueError('Wrong active recovery schedule')
    expected = dict(action_gt_weight=1., teacher_action_weight=0., teacher_video_weight=.05,
                    kv_weight=0., kv_layers=[], frozen_video=False, frozen_action=True)
    if any(config['train'].get(k) != v for k,v in expected.items()):
        raise ValueError('Incorrect no-K/V objective or frozen-module setting')


def original_recipe(config):
    """Only approved recipe differences are removed for parent provenance checks."""
    c = copy.deepcopy(config)
    c.pop('recovery_lr_branch', None)
    c.pop('recovery_extension', None)
    c['recovery'] = dict(updates=2344, learning_rate=5e-6, warmup_updates=250, final_lr_fraction=.1)
    c['train']['kv_weight'] = .05
    c['train']['kv_layers'] = list(range(20,40))
    return c


def learning_rate(config, update):
    validate(config)
    p = config['recovery_lr_branch']
    if not p['parent_update'] < update <= p['schedule_endpoint']:
        raise ValueError('LR requested outside continuation')
    if update <= p['hold_through']:
        return p['learning_rate']
    progress = (update-p['hold_through'])/(p['schedule_endpoint']-p['hold_through'])
    return p['learning_rate']*(p['final_lr_fraction']+(1-p['final_lr_fraction'])*.5*(1+math.cos(math.pi*progress)))


def verify_parent(config):
    from .recovery_subset import validate_parent
    from vpp2.full_mask.panel import file_hash
    validate(config)
    _, parent_receipt = validate_parent(config)
    p = config['recovery_lr_branch']; root = Path(p['comparison_root'])
    if file_hash(p['baseline55_validation']) != p['baseline55_validation_sha256']:
        raise ValueError('Checkpoint55 validation changed')
    for name, expected in p['comparison_pins'].items():
        if file_hash(root/name) != expected:
            raise ValueError('Changed historical comparison artifact: '+name)
    old = json.loads((root/'config.json').read_text())
    expected = original_recipe(config)
    for c in (expected, old):
        c.pop('root'); c['wandb'].pop('name'); c['wandb'].pop('group')
    if expected != old:
        raise ValueError('Unapproved difference from the matched original recipe')
    native = root/'recovery/diagnostics/native_000250.json'
    if json.loads(native.read_text())['update'] != 250:
        raise ValueError('Wrong historical comparison')
    # The original250 weights were never saved. Do not require or pretend that
    # recorded validation data are a restored checkpoint.
    return parent_receipt, dict(historical_validation=str(native),
        validation_sha256=file_hash(native),
        historical_checkpoint250_available=False,
        paused_replay_root=p['paused_replay_root'],
        protected_from_child_retention=True, used_as_training_initialization=False)


def check_storage(config):
    p = config['recovery_lr_branch']; root = Path(config['root'])
    storage = Path(p['artifact_storage'])
    for name in ('checkpoints','best_models','exports'):
        link = root/'recovery'/name
        if not link.is_symlink() or link.resolve() != (storage/name).resolve():
            raise ValueError('Wrong artifact storage: '+name)
    if Path(p['checkpoint_storage']).resolve() != (storage/'checkpoints').resolve():
        raise ValueError('Wrong restart storage')
    # Reserve the total run footprint, including existing files on a resume.
    occupied = sum(f.stat().st_size for f in storage.rglob('*') if f.is_file() and not f.is_symlink())
    if shutil.disk_usage(storage).free + occupied < 650*2**30:
        raise RuntimeError('Need a 650 GiB run budget for restart states, policy exports, and temporary loading')
    if shutil.disk_usage(root).free < 5*2**30:
        raise RuntimeError('Need 5 GiB for source, logs and diagnostics')


def state_fields(config):
    from vpp2.full_mask.panel import digest
    return {'recovery_lr_branch_identity':digest(config['recovery_lr_branch'])}


def check_state(config, state):
    if state is None or state.get('recovery_lr_branch_identity') != state_fields(config)['recovery_lr_branch_identity']:
        raise ValueError('Wrong no-K/V continuation checkpoint')
    if not 55 <= state['update'] <= config['recovery_lr_branch']['schedule_endpoint']:
        raise ValueError('Invalid branch update')


def load_parent(runtime, config):
    import torch.distributed as dist
    from . import recovery_subset
    from .recovery_best import candidate, verify_artifact
    from vpp2.compact_kv.spec import atomic_json
    from vpp2.full_mask.panel import file_hash
    receipt, binding = verify_parent(config)
    p = config['recovery_lr_branch']
    state = recovery_subset.load_parent(runtime, config)
    if state['update'] != 55 or state['cursor'] != 14080 or state['recovery_subset_cursor'] != 0:
        raise ValueError('Checkpoint55 optimizer or curated sample cursor was not preserved')
    validation = Path(p['baseline55_validation'])
    item = candidate(json.loads(validation.read_text()), runtime.physical_parameters, validation)
    checkpoint_path = Path(p['parent_root'])/'recovery/checkpoints/update_000055'
    item['artifact'] = dict(kind='training_checkpoint',path=str(checkpoint_path),
        receipt_sha256=file_hash(checkpoint_path/'COMPLETE.json'),checkpoint_identity=receipt['checkpoint_identity'],
        initial_export_sha256=receipt['recovery_export_sha256'])
    verify_artifact(item)
    # Historical250 is an external comparator, never a future candidate in this run.
    state = dict(state,best=[item],protected_replay_checkpoint=binding,**state_fields(config))
    check_state(config,state)
    if runtime.rank == 0:
        atomic_json(Path(config['root'])/'control/protected_historical_references.json',binding)
    dist.barrier()
    return state


def verify_restored_validation(runtime, result):
    from vpp2.compact_kv.spec import atomic_json
    p = Path(runtime.config['recovery_lr_branch']['baseline55_validation'])
    old = json.loads(p.read_text()); before = {r['window_id']:r for r in old['rows']}
    if result['update'] != 55 or set(before) != {r['window_id'] for r in result['rows']}:
        raise ValueError('Restored validation panel or update changed')
    for r in result['rows']:
        b = before[r['window_id']]
        for k in ('student_error','teacher_error'):
            if not math.isclose(r[k],b[k],rel_tol=.001,abs_tol=1e-8):
                raise ValueError('Restored checkpoint55 validation mismatch')
        for k in ('student_components','teacher_components'):
            if any(not math.isclose(x,y,rel_tol=.001,abs_tol=1e-8) for x,y in zip(r[k],b[k],strict=True)):
                raise ValueError('Restored checkpoint55 component mismatch')
    if runtime.rank == 0:
        atomic_json(Path(runtime.config['root'])/'checks/restore55.json',dict(passed=True,update=55,
            full_panel_windows=140,reference_checkpoint_unchanged=True,
            no_kv_training_starts_at_update=56))


def validate_forward_storage(runtime):
    if runtime.masks is not None or runtime.mask_optimizer is not None:
        raise RuntimeError('Recovery must not train masks')
    runtime.check_frozen()
    if not all(p.requires_grad for p in runtime.engine.module.parameters()):
        raise RuntimeError('Expected the compact video weights to be trainable')
