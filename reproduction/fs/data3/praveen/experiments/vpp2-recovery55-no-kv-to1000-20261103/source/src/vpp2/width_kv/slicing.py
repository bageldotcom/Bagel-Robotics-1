"""Explicit, audited tensor transfer; residual coordinates differ from attention coordinates."""
import json
from pathlib import Path

import torch

from vpp2.models.wan21_14b.wan_video_dit import WanVideoDiT, precompute_freqs_cis_3d
from vpp2.compact_kv.blocks import install_blocks, TerminalCacheBlock
from vpp2.compact_kv.spec import file_sha


def evenly_spaced(total, keep):
    if not 0 < keep <= total:
        raise ValueError('Invalid retained channel count')
    indices = torch.linspace(0, total - 1, keep, dtype=torch.float64).round().long().tolist()
    if len(set(indices)) != keep:
        raise AssertionError('Repeated channel index')
    return indices


def validate_indices(indices, total, keep):
    if len(indices) != keep or indices != sorted(set(indices)) or any(type(i) is not int or not 0 <= i < total for i in indices):
        raise ValueError('Channel indices must be sorted, unique, in range and complete')


def standard_selection(width=2048, ffn=8192, teacher_width=5120, teacher_ffn=13824, layers=40):
    return dict(schema='vpp2-width-selection-v1', method='evenly_spaced',
                hidden_indices=evenly_spaced(teacher_width, width),
                ffn_indices=[evenly_spaced(teacher_ffn, ffn) for _ in range(layers)])


def slice_tensor(name, value, hidden, ffn, teacher_width):
    """Return a clone and a human-auditable rule for every native parameter."""
    h = torch.tensor(hidden, dtype=torch.long)
    rows = cols = None
    rule = None
    if name.startswith('blocks.'):
        _, block, rest = name.split('.', 2)
        f = torch.tensor(ffn[int(block)], dtype=torch.long)
        if rest == 'modulation':
            return value.index_select(-1, h).clone(), 'residual modulation coordinates'
        if rest.startswith(('self_attn.', 'cross_attn.')):
            pieces = rest.split('.')
            projection, kind = pieces[1:]
            if projection in ('norm_q', 'norm_k', 'norm_k_img'):
                rule = 'retain all attention normalization coordinates'
            elif projection in ('q', 'k', 'v', 'k_img', 'v_img'):
                cols = h if kind == 'weight' else None
                rule = 'retain attention outputs; select residual inputs'
            elif projection == 'o':
                rows = h
                rule = 'select residual outputs; retain attention inputs'
        elif rest.startswith('norm3.'):
            rows, rule = h, 'residual affine normalization'
        elif rest.startswith('ffn.0.'):
            rows, cols, rule = f, h if rest.endswith('weight') else None, 'FFN input projection'
        elif rest.startswith('ffn.2.'):
            rows, cols, rule = h, f if rest.endswith('weight') else None, 'FFN output projection'
    elif name.startswith('patch_embedding.'):
        rows, rule = h, 'patch embedding output channels'
    elif name.startswith(('text_embedding.0.', 'time_embedding.0.')):
        rows, rule = h, 'conditioning projection output channels'
    elif name.startswith(('text_embedding.2.', 'time_embedding.2.')):
        rows, cols, rule = h, h if name.endswith('weight') else None, 'conditioning hidden coordinates'
    elif name.startswith('time_projection.1.'):
        rows = torch.cat([h + group * teacher_width for group in range(6)])
        cols = h if name.endswith('weight') else None
        rule = 'six distinct timestep modulation groups'
    elif name == 'head.modulation':
        return value.index_select(-1, h).clone(), 'output head modulation coordinates'
    elif name.startswith('head.head.'):
        cols, rule = h if name.endswith('weight') else None, 'retain native video output channels'
    elif name.startswith(('img_emb.proj.0.', 'img_emb.proj.1.')):
        rule = 'retain CLIP input-coordinate transform'
    elif name.startswith(('img_emb.proj.3.', 'img_emb.proj.4.')):
        rows, rule = h, 'image conditioning residual coordinates'
    if rule is None:
        raise ValueError('Unaccounted tensor: ' + name)
    if rows is not None:
        value = value.index_select(0, rows)
    if cols is not None:
        value = value.index_select(1, cols)
    return value.clone(), rule


def slice_state(teacher, student_shapes, selection, teacher_width=5120):
    if set(teacher) != set(student_shapes):
        raise ValueError('Teacher/student tensor inventories differ')
    hidden, ffns = selection['hidden_indices'], selection['ffn_indices']
    states, audit = {}, {}
    for name, shape in student_shapes.items():
        value, rule = slice_tensor(name, teacher[name], hidden, ffns, teacher_width)
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f'{name}: sliced {tuple(value.shape)} != required {tuple(shape)}')
        states[name] = value
        audit[name] = dict(source_shape=list(teacher[name].shape), target_shape=list(shape), rule=rule)
    return states, audit


def load_width_student(config, candidate):
    from omegaconf import OmegaConf
    from vpp2.compact_kv.model import native_config, VideoBranch
    if candidate not in ('W1', 'W2'):
        raise ValueError(candidate)
    selection_path = Path(config['root']) / 'calibration' / f'{candidate}_selection.json'
    selection = json.loads(selection_path.read_text())
    if selection['candidate'] != candidate:
        raise ValueError('Wrong channel-selection identity')
    validate_indices(selection['hidden_indices'], 5120, 2048)
    if len(selection['ffn_indices']) != 40:
        raise ValueError('Expected all 40 FFN selections')
    for indices in selection['ffn_indices']:
        validate_indices(indices, 13824, 8192)
    native = native_config(config)
    kwargs = OmegaConf.to_container(native.model.video_dit_config, resolve=True)
    kwargs.update(dim=2048, hidden_dim=2048, ffn_dim=8192, num_heads=40, attn_head_dim=128, num_layers=40)
    with torch.device('meta'):
        video = WanVideoDiT(**kwargs)
    source = Path(config['teacher_bundle']) / 'video.pt'
    teacher = torch.load(source, map_location='cpu', weights_only=True, mmap=True)['dit']
    states, audit = slice_state(teacher, {k: tuple(v.shape) for k, v in video.state_dict().items()}, selection)
    video.load_state_dict(states, strict=True, assign=True)
    video.freqs = precompute_freqs_cis_3d(128)
    install_blocks(video, ())
    video.requires_grad_(True)
    video.head.requires_grad_(False)
    # Final post-cache state updates cannot affect any action-facing cache.
    # Retain their weights in the architecture/export, but exclude them from AdamW.
    final = video.blocks[-1]
    for module in (final.self_attn.q, final.self_attn.norm_q, final.self_attn.o,
                   final.cross_attn, final.ffn, final.norm3):
        module.requires_grad_(False)
    # Preserve all native tensors, omitting only the unconsumed final video update.
    final.__class__ = TerminalCacheBlock
    video.use_gradient_checkpointing = True
    branch = VideoBranch(video)
    branch.width_transfer_audit = dict(candidate=candidate, selection_sha256=file_sha(selection_path),
                                     tensors=audit, parameters=sum(p.numel() for p in video.parameters()),
                                     trainable_parameters=sum(p.numel() for p in video.parameters() if p.requires_grad))
    return branch
