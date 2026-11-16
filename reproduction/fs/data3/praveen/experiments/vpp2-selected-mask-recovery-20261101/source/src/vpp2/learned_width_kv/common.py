"""Experiment identities, native model loading and isolated input manifests."""
import copy
import json
import importlib.metadata
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from vpp2.compact_kv.model import native_config, VideoBranch
from vpp2.compact_kv.blocks import install_blocks, TerminalCacheBlock
from vpp2.compact_kv.spec import file_sha
from vpp2.models.wan21_14b.wan_video_dit import WanVideoDiT, precompute_freqs_cis_3d
from .masking import install_masks


def source_hashes():
    repo = Path(__file__).resolve().parents[3]
    files = list((repo/'src/vpp2').rglob('*.py')) + list((repo/'scripts/learned_width_kv').rglob('*')) + list((repo/'tests/learned_width_kv').rglob('*.py'))
    files += [p for p in (repo/'configs').rglob('*') if p.suffix in ('.yaml','.yml','.json')]
    return {str(p.relative_to(repo)):file_sha(p) for p in sorted(set(files)) if p.is_file() and '__pycache__' not in p.parts}


def validate_recipe(config):
    # Several architecture/schedule choices are deliberately fixed in this pilot.
    # Reject a config-only change instead of silently reporting an unused value.
    fixed = dict(global_batch=288, mask_updates=1000, recovery_updates=1000,
        max_updates=1000, warmup_updates=50, lr_final_fraction=.1,
        action_gt_weight=1., action_teacher_weight=1., kv_max_weight=.05,
        kv_ramp_updates=100, mask_final_lr_fraction=.1, mask_weight_decay=0.,
        mask_clip=1., checkpoint_every=250, validation_every=100,
        full_validation_updates=[0,250,500,750,1000], profile_updates=6,
        profile_discard=2, profile_reserved_limit_gib=72., strict_gpu_count=8,
        zero_stage=3, shard_frozen_teacher=True, teacher_supervision_gate="shared_action_and_kv_v1")
    for name, expected in fixed.items():
        if config.get(name) != expected:
            raise ValueError(f'Fixed pilot recipe mismatch: {name} must be {expected!r}')


def kv_coefficient(config, phase, update):
    """Phase-local KV schedule; recovery starts at the full configured weight."""
    if phase not in ('A', 'B') or update < 1:
        raise ValueError('KV schedule requires phase A/B and a positive update')
    ramp = min(update / config['kv_ramp_updates'], 1.) if phase == 'A' else 1.
    return config['kv_max_weight'] * ramp


def verify_environment(inputs):
    if inputs['python'] != sys.version:
        raise RuntimeError('Pinned Python runtime changed')
    for name, expected in inputs['environment'].items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError('Pinned package version changed: ' + name)


def identity(config):
    validate_recipe(config)
    root = Path(config['root'])
    inputs=json.loads((root/'inputs.json').read_text())
    verify_environment(inputs)
    if file_sha(root/'calibration/initial_selection.json')!=inputs['selection_sha256']:
        raise RuntimeError('Pinned initial channel selection changed')
    for name,digest in inputs['data'].items():
        if file_sha(root/'data'/name)!=digest:raise RuntimeError('Pinned data manifest changed: '+name)
    for name,expected in inputs['teacher_stat'].items():
        current=(Path(config['teacher_bundle'])/name).stat()
        if [current.st_size,current.st_mtime_ns]!=expected:raise RuntimeError('Pinned teacher file changed: '+name)
    return dict(schema='learned-width-identity-v1', configuration=config, sources=source_hashes(),inputs=inputs)


def verify_receipt(path, expected):
    receipt = json.loads(Path(path).read_text())
    if not receipt.get('passed'):
        raise RuntimeError('Missing or failed readiness receipt: '+str(path))
    if receipt.get('identity') != expected:
        from .performance import verify_performance_receipt
        verify_performance_receipt(path, receipt, expected)
    return receipt


def build_video(config, compact=False):
    kwargs = OmegaConf.to_container(native_config(config).model.video_dit_config, resolve=True)
    if compact:
        kwargs.update(dim=2048,hidden_dim=2048,ffn_dim=8192,num_layers=40,num_heads=40,attn_head_dim=128)
    with torch.device('meta'):
        video = WanVideoDiT(**kwargs)
    return video


def freeze_unused(video):
    video.requires_grad_(True)
    video.head.requires_grad_(False)
    final = video.blocks[-1]
    for module in (final.self_attn.q,final.self_attn.norm_q,final.self_attn.o,final.cross_attn,final.ffn,final.norm3):
        module.requires_grad_(False)
    video.use_gradient_checkpointing = True


def load_student(config, phase):
    compact = phase == 'B'
    video = build_video(config,compact)
    if phase == 'A':
        source = Path(config['teacher_bundle'])/'video.pt'
    elif phase == 'B':
        source = Path(config['root'])/'surgery/native/video.pt'
        receipt = verify_receipt(Path(config['root'])/'checks/surgery.json',identity(config))
        if file_sha(source)!=receipt['video_sha256']:
            raise RuntimeError('Compact initialization changed after verified surgery')
    else:
        raise ValueError(phase)
    state = torch.load(source,map_location='cpu',mmap=True,weights_only=True)['dit']
    # Cloning prevents any writable alias to mmap teacher storage.
    video.load_state_dict({k:v.clone() for k,v in state.items()},strict=True,assign=True)
    video.freqs = precompute_freqs_cis_3d(128)
    freeze_unused(video)
    if not compact:
        return install_masks(video)
    install_blocks(video,())
    video.blocks[-1].__class__ = TerminalCacheBlock
    return VideoBranch(video)


def phase_config(config, phase, micro):
    result = copy.deepcopy(config)
    result['micro_batch'] = micro
    result['learning_rate'] = config['mask_phase_video_lr'] if phase=='A' else config['recovery_video_lr']
    return result


def architecture(selection):
    return dict(schema='vpp2-compact-width-v1',candidate='learned-mask',native_depth=40,
                kv_only_indices=[],hidden_dim=2048,ffn_dim=8192,num_heads=40,head_dim=128,
                temporal_planes=5,tokens_per_plane=390,selection=selection,inference_teacher_required=False)
