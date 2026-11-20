"""Final-method configuration and canonical source/configuration identity."""
import copy
import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def load(path):
    config=json.loads(Path(path).read_text())
    validate(config)
    if config.get('root'):config.setdefault('curation',{}).setdefault('output',str(Path(config['root'])/'selection'))
    return config


def validate(config):
    if config.get('schema')!='bagel-method-v1':raise ValueError('Unsupported method configuration')
    if config.get('stage') not in ('mask','recovery'):raise ValueError('Expected mask or recovery stage')
    if type(config.get('seed')) is not int:raise ValueError('An integer seed is required')
    train=config['train'];geometry=config['geometry']
    if (train['global_batch'],train['world_size'])!=(256,8):raise ValueError('The distributed recipe uses batch 256 on eight GPUs')
    if train['micro_batch'] not in (1,2,4) or train['global_batch']%(train['world_size']*train['micro_batch']):raise ValueError('Invalid microbatch/accumulation geometry')
    if not train['frozen_action'] or train['frozen_video'] or train['teacher_action_weight']!=0:
        raise ValueError('Train the video expert through the fixed action expert')
    if train['action_gt_weight']!=1. or train['teacher_video_weight']!=.05 or train['kv_weight'] not in (0.,.05):
        raise ValueError('Unsupported objective coefficients')
    for key in ('video_lr','mask_lr','gradient_clip'):
        if not math.isfinite(train[key]) or train[key]<=0:raise ValueError('Invalid optimizer setting: '+key)
    if train['betas']!=[.9,.999] or train['video_weight_decay']!=.01 or train['mask_weight_decay']!=0.:
        raise ValueError('Optimizer convention changed')
    if type(train['max_updates']) is not int or train['max_updates']<1:raise ValueError('A positive training endpoint is required')
    if (geometry['layers'],geometry['hidden'],geometry['ffn'],geometry['heads'],geometry['head_dim'])!=(40,5120,13824,40,128):
        raise ValueError('The native model geometry changed')
    for axis,floor in [('hidden','hidden_floor'),('ffn','ffn_floor'),('heads','head_floor')]:
        if not 0<geometry[floor]<=geometry[axis]:raise ValueError('Invalid retained-dimension bound')
    if geometry['pack']!=128:raise ValueError('Native channel packing must remain 128')
    native=config['native_inference']
    if (native['steps'],native['horizon'],native['replan'],native['history_stride'],native['shift'])!=(10,32,24,25,1.):
        raise ValueError('Native training diagnostic contract changed')
    if config['validation']!={'fp32_relative_tolerance':2e-5,'bf16_action_video_relative_tolerance':.01}:
        raise ValueError('Export numerical tolerances changed')
    size=config['size_coefficient']
    if not math.isfinite(size['upper_cap']) or not 0<size['upper_cap']<=1.:raise ValueError('Invalid size coefficient cap')
    value=size.get('initial_value')
    if value is not None and (not math.isfinite(value) or not 0<value<=size['upper_cap']):raise ValueError('Invalid initial size coefficient')
    if size.get('pressure_multiplier',1.)<1 or size.get('pressure_ramp',1)<1:raise ValueError('Invalid size-penalty schedule')
    recovery=config['recovery']
    if not 0<recovery['learning_rate'] or not 0<recovery['final_lr_fraction']<=1:raise ValueError('Invalid recovery learning rate')
    if not 0<=recovery['hold_updates']<recovery['schedule_updates']:raise ValueError('Invalid recovery schedule horizon')
    if recovery.get('sample_mode','selected_windows') not in ('selected_windows','all_episode_windows'):raise ValueError('Invalid recovery sample mode')
    for name in ('validation_every','checkpoint_every'):
        if type(recovery[name]) is not int or recovery[name]<1:raise ValueError('Invalid recovery interval')


def sources():
    package=Path(__file__).resolve().parents[1]
    return {str(path.relative_to(package)):hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ('adaptive_pruning','recovery','models','_internal','data_selection')
            for path in sorted((package/folder).rglob('*.py')) if path.name!='provenance.py'}


def identity(config):
    resolved=copy.deepcopy(config)
    # Checkpoints can be moved to a new output directory without changing their
    # training semantics. Asset identities and all numerical settings remain bound.
    for name in ('root','resume','runtime_conditioning_cache','demonstrations','teacher_bundle','wan_root','panel_file','initial_masks','compact_model'):
        resolved.pop(name,None)
    resolved.get('curation',{}).pop('output',None)
    return digest(dict(config=resolved,sources=sources()))
