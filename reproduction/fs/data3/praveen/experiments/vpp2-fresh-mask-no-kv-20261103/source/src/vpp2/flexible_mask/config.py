"""Strict recipe and content identity; old frozen-capacity controllers fail closed."""
import json
import math
from pathlib import Path
from vpp2.full_mask.panel import file_hash, digest


def load(path):
    c=json.loads(Path(path).read_text())
    validate(c)
    return c


def validate(c):
    fixed = {'schema':'vpp2-flexible-joint-mask-v1','seed':100008,
             'original_video_parameters':16394878784}
    for k,v in fixed.items():
        if c.get(k)!=v: raise ValueError('Recipe mismatch: '+k)
    for key in ('budget','capacity_feedback','hidden_search','mask_priority','quality_gate'):
        if key in c: raise ValueError('Legacy frozen capacity control is forbidden: '+key)
    expected = dict(global_batch=256,world_size=8,video_lr=1e-6,mask_lr=.01,
        betas=[.9,.999],video_weight_decay=.01,mask_weight_decay=0.,gradient_clip=1.,
        warmup_updates=50,max_updates=3000,minimum_updates=2000,final_lr_fraction=.1,
        action_gt_weight=1.,teacher_action_weight=0.,teacher_video_weight=.05,kv_weight=.05,
        kv_layers=list(range(20,40)),count_budget_weight=0.,frozen_video=False,frozen_action=True,
        checkpoint_every=250,flow_monitor_every=500,native_windows_per_task=4)
    if c.get("regularization_replay"):
        expected["kv_weight"]=0.
    for k,v in expected.items():
        if c['train'].get(k)!=v: raise ValueError('Training recipe mismatch: '+k)
    if type(c['train'].get('micro_batch')) is not int or c['train']['micro_batch'] not in (1,2,4):
        raise ValueError('Only microbatch candidates one, two and four are supported; each needs GPU verification')
    expected_size=dict(target_gradient_fraction=.1,first_update=193,last_regular_update=200,last_replacement_update=232,samples_per_family=8,ramp_updates=100,invalid_measurement='checkpoint_and_stop')
    if {k:v for k,v in c['size_coefficient'].items() if k!='upper_cap'}!=expected_size:
        raise ValueError('Size calibration recipe changed')
    if c['initial_preferred']!=dict(hidden_scores=4864,ffn_scores=9728,cross_scores=32):
        raise ValueError('Initial ranking preference changed')
    if c['validation']!=dict(fp32_relative_tolerance=2e-5,bf16_action_video_relative_tolerance=.01):
        raise ValueError('Numerical verification tolerance changed')
    cap=c['size_coefficient']['upper_cap']
    if not isinstance(cap,(int,float)) or not math.isfinite(cap) or not 0<cap<=1.:
        raise ValueError('Size coefficient cap must be in (0,1]')
    if c['reliability']['method']!='relative_student_advantage_with_baseline_component_veto':
        raise ValueError('Old reliability base is forbidden')
    if c['geometry']['hidden_floor']!=4864 or c['geometry']['ffn_floor']!=4096 or c['geometry']['head_floor']!=8:
        raise ValueError('Search bounds changed')
    execution=c.get('execution',{})
    if (set(execution)-{'teacher_storage','fill_uninitialized_memory','cached_norm_flags','skip_nograd_checkpointing','mixed_attention_checkpointing','deduplicate_conditioning_reads','calibration_micro_batch'} or
        execution.get('teacher_storage','sharded') not in ('sharded','replicated') or
        type(execution.get('fill_uninitialized_memory',True)) is not bool):
        raise ValueError('Unverified execution recipe')
    for key in ('cached_norm_flags','skip_nograd_checkpointing','mixed_attention_checkpointing','deduplicate_conditioning_reads'):
        if key in execution and type(execution[key]) is not bool:raise ValueError('Execution switches must be boolean: '+key)
    if 'calibration_micro_batch' in execution and (type(execution['calibration_micro_batch']) is not int or execution['calibration_micro_batch']!=2):
        raise ValueError('Only two-window calibration subbatches are verified candidates')
    if c['recovery']!=dict(updates=6000,learning_rate=5e-6,warmup_updates=250,final_lr_fraction=.1):
        raise ValueError('Recovery recipe changed')
    from .pressure import validate as validate_pressure
    validate_pressure(c)
    from .replay import validate as validate_replay
    validate_replay(c)


def sources():
    repo=Path(__file__).resolve().parents[3]
    return {str(p.relative_to(repo)):file_hash(p) for base in ('src','scripts','tests','configs')
            for p in sorted((repo/base).rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.yaml','.sh','.sbatch')
            and '__pycache__' not in p.parts}


def identity(config):
    return digest(dict(config=config,sources=sources()))
