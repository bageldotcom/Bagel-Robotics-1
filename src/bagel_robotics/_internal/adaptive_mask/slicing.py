"""Exact hidden/FFN tensor slicing in original channel coordinates."""
import torch

def validate_indices(indices, total, keep):
    if len(indices) != keep or indices != sorted(set(indices)) or any(type(i) is not int or not 0 <= i < total for i in indices):
        raise ValueError('Channel indices must be sorted, unique, in range and complete')

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
