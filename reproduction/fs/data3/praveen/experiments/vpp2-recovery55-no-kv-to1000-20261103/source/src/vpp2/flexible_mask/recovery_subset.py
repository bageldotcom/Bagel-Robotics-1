"""Explicit dataset-phase continuation: exact curated windows, pinned parent state."""
import copy
import json
from pathlib import Path
from vpp2.full_mask.panel import file_hash, digest

SCHEMA = 'curated30k-recovery-continuation-v1'


def validate_settings(config):
    phase=config.get('recovery_subset')
    if phase is None:return
    required={'schema','sampler','windows','updates','parent_root','parent_update',
        'parent_identity','parent_tag','parent_config_sha256','parent_sources_sha256',
        'parent_inputs_sha256','parent_receipt_sha256','parent_checkpoint_identity',
        'parent_export_receipt_sha256','parent_selected_sha256','export_manifest_sha256',
        'export_path'}
    if set(phase)!=required:raise ValueError('Incomplete recovery subset provenance')
    if (phase['schema']!=SCHEMA or phase['sampler']!='original_curated_task_balanced'
        or phase['windows']!=30000 or phase['updates']!=2344):
        raise ValueError('Unsupported recovery subset recipe')
    start=phase['parent_update']
    if type(start) is not int or not 0<start<6000:raise ValueError('Invalid parent update')
    if phase['parent_tag']!=f'update_{start:06d}':raise ValueError('Invalid parent tag')
    for key in ('parent_root','export_path'):
        if not Path(phase[key]).is_absolute():raise ValueError('Parent paths must be absolute')
    if Path(config['root']).resolve()==Path(phase['parent_root']).resolve():
        raise ValueError('Dataset phase requires a separate root')
    if config.get('resume_from') or config.get('pressure_continuation'):
        raise ValueError('Cannot combine recovery dataset and mask continuations')
    if config['recovery']['updates']!=phase['updates']:raise ValueError('Subset endpoint mismatch')


def start_update(config):
    return config.get('recovery_subset',{}).get('parent_update',0)


def endpoint(config):
    return start_update(config)+config['recovery']['updates']


def sample_position(config,position):
    result=position-start_update(config)*config['train']['global_batch']
    if result<0:raise ValueError('Sample precedes curated recovery phase')
    return result


def state_fields(config,update):
    if not config.get('recovery_subset'):return {}
    elapsed=update-start_update(config)
    if not 0<=elapsed<=config['recovery']['updates']:raise ValueError('Update outside dataset phase')
    return dict(recovery_subset_cursor=elapsed*config['train']['global_batch'],
        recovery_subset_parent=config['recovery_subset']['parent_checkpoint_identity'])


def check_state(config,state):
    if state is None or not config.get('recovery_subset'):return
    expected=state_fields(config,state['update'])
    if any(state.get(k)!=v for k,v in expected.items()):
        raise ValueError('Wrong curated recovery cursor or parent')


def validate_recipe(config,old):
    if config.get('recovery_lr_branch'):
        from .low_lr import original_recipe
        config=original_recipe(config)
    left=copy.deepcopy(config);right=copy.deepcopy(old)
    left.pop('recovery_subset',None)
    for c in (left,right):
        c.pop('root')
        c['wandb'].pop('name',None);c['wandb'].pop('group',None)
        c['recovery'].pop('updates')
    if left!=right:raise ValueError('Dataset continuation changed unrelated training behavior')


def validate_parent(config):
    validate_settings(config)
    s=config['recovery_subset'];root=Path(s['parent_root'])
    entries={
        'config.json':'parent_config_sha256','source_manifest.json':'parent_sources_sha256',
        'inputs.json':'parent_inputs_sha256','export_verified.json':'parent_export_receipt_sha256',
        'selected.json':'parent_selected_sha256',
        f"recovery/checkpoints/{s['parent_tag']}/COMPLETE.json":'parent_receipt_sha256'}
    for name,key in entries.items():
        if file_hash(root/name)!=s[key]:raise ValueError('Parent artifact changed: '+name)
    old=json.loads((root/'config.json').read_text());validate_recipe(config,old)
    sources=json.loads((root/'source_manifest.json').read_text())
    if (sources['identity']!=s['parent_identity'] or
        digest(dict(config=old,sources=sources['files']))!=s['parent_identity']):
        raise ValueError('Parent source/config identity mismatch')
    # The parent is preserved; source hashes and immutable receipts bind its origin.
    receipt=json.loads((root/f"recovery/checkpoints/{s['parent_tag']}/COMPLETE.json").read_text())
    without_identity={k:v for k,v in receipt.items() if k!='checkpoint_identity'}
    if (receipt['checkpoint_identity']!=digest(without_identity) or
        receipt['checkpoint_identity']!=s['parent_checkpoint_identity'] or
        receipt['identity']!=s['parent_identity'] or receipt['stage']!='recovery' or
        receipt['update']!=s['parent_update'] or
        receipt['recovery_export_sha256']!=s['export_manifest_sha256']):
        raise ValueError('Parent recovery checkpoint mismatch')
    export=json.loads((root/'export_verified.json').read_text())
    if (export['identity']!=s['parent_identity'] or export.get('passed') is not True or
        export['path']!=s['export_path'] or
        file_hash(Path(s['export_path'])/'manifest.json')!=s['export_manifest_sha256']):
        raise ValueError('Parent compact export changed')
    for key in ('panel_file','panel_sha256','normalization_sha256','teacher_bundle'):
        if config[key]!=old[key]:raise ValueError('Model/data artifact changed')
    return old,receipt


def require_export(config,export):
    from .recovery_handoff import require_selected
    if not config.get('recovery_subset'):
        return require_selected(config,export)
    old,_=validate_parent(config)
    s=config['recovery_subset']
    if str(Path(export).resolve())!=str(Path(s['export_path']).resolve()):
        raise ValueError('Wrong compact initialization export')
    return require_selected(old,export,expected_identity=s['parent_identity'])


def load_parent(runtime,config):
    from . import checkpoint
    from .config import identity
    old,receipt=validate_parent(config);s=config['recovery_subset']
    state=checkpoint.load(runtime,Path(old['root'])/'recovery',s['parent_identity'],
        'recovery',tag=s['parent_tag'])
    if state['update']!=s['parent_update']:raise ValueError('Incorrect parent optimizer update')
    # Retain optimizer, RNG and global step; restart only the new dataset cursor.
    state=dict(state,identity=identity(config),parent_checkpoint_identity=receipt['checkpoint_identity'],
        parent_history=state['history'],parent_best=state['best'],history=[],best=[],
        **state_fields(config,state['update']))
    check_state(config,state)
    return state
