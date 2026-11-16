"""Explicit, content-pinned continuation across validated execution changes."""
import copy
import json
from pathlib import Path
from . import checkpoint
from .config import identity
from vpp2.full_mask.panel import file_hash


def scientific_recipe(config):
    result=copy.deepcopy(config)
    for key in ('root','resume_from'):
        result.pop(key,None)
    result['train'].pop('micro_batch')
    result.setdefault('execution',{}).pop('fill_uninitialized_memory',None)
    if not result['execution']:result.pop('execution')
    result['wandb'].pop('name',None)
    return result


def validate_parent(config,stage):
    source=config['resume_from']
    if stage!='mask':raise ValueError('This execution migration applies only to mask learning')
    parent=Path(source['root'])
    if file_hash(parent/'config.json')!=source['config_sha256']:
        raise ValueError('Parent configuration changed')
    old=json.loads((parent/'config.json').read_text())
    if scientific_recipe(config)!=scientific_recipe(old):
        raise ValueError('Execution continuation changed the scientific recipe')
    receipt_path=parent/'mask/checkpoints'/source['tag']/'COMPLETE.json'
    if file_hash(receipt_path)!=source['receipt_sha256']:
        raise ValueError('Parent checkpoint receipt changed')
    receipt=json.loads(receipt_path.read_text())
    if receipt['identity']!=source['identity'] or receipt['stage']!=stage:
        raise ValueError('Wrong parent checkpoint identity/stage')
    if Path(source['tag']).name!=source['tag'] or source['tag']!=f"update_{receipt['update']:06d}":
        raise ValueError('Invalid parent checkpoint tag')
    old_inputs=json.loads((parent/'inputs.json').read_text())
    current_inputs=json.loads((Path(config['root'])/'inputs.json').read_text())
    old_inputs.pop('identity');current_inputs.pop('identity')
    if old_inputs!=current_inputs:raise ValueError('Continuation data/model inputs changed')
    return parent,receipt


def load_parent(runtime,config,stage):
    parent,receipt=validate_parent(config,stage)
    state=checkpoint.load(runtime,parent/'mask',receipt['identity'],stage,tag=config['resume_from']['tag'])
    state=dict(state,identity=identity(config),parent_checkpoint_identity=receipt['checkpoint_identity'])
    return state
