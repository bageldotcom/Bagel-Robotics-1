"""Approved export checks: strict FP32 structure and BF16 prediction outputs.

BF16 cache differences remain reported for every layer. They are not silently
treated as zero or as evidence of policy retention. The final physical-policy
audit is separate from this numerical implementation check.
"""
import math
import os
import torch


POLICY = 'fp32_structure_bf16_outputs_v1'


def compare(actual, expected, name):
    if actual.shape != expected.shape:
        raise ValueError('Equivalence shape mismatch: ' + name)
    difference = actual.float() - expected.float()
    scale = expected.float().square().mean().sqrt().clamp_min(1e-6)
    return dict(tensor=name, max_absolute=float(difference.abs().max()),
                relative_rmse=float(difference.square().mean().sqrt() / scale))


def assess(fp32, bf16, layers, fp32_limit, output_limit):
    """Fail closed on incomplete, duplicate, nonfinite or excessive differences."""
    expected_fp32 = {'embedding_' + k for k in ('tokens', 't', 't_mod', 'context')} | {'video'}
    expected_fp32 |= {f'layer_{i:02d}_{kind}' for i in range(layers) for kind in ('k', 'v', 'hidden')}
    expected_bf16 = {'video', 'action'} | {f'layer_{i:02d}_{kind}' for i in range(layers) for kind in ('k', 'v')}
    failures = []
    for label, rows, expected in [('fp32', fp32, expected_fp32), ('bf16', bf16, expected_bf16)]:
        names = [r['tensor'] for r in rows]
        if set(names) != expected or len(names) != len(expected):
            failures.append(label + '_incomplete_or_duplicate_comparisons')
        if any(not math.isfinite(r[k]) or r[k] < 0 for r in rows for k in ('relative_rmse', 'max_absolute')):
            failures.append(label + '_nonfinite_or_invalid_difference')
    if not (math.isfinite(fp32_limit) and fp32_limit > 0 and math.isfinite(output_limit) and output_limit > 0):
        raise ValueError('Invalid equivalence tolerance')
    fp32_max = max((r['relative_rmse'] for r in fp32), default=float('inf'))
    if fp32_max > fp32_limit:
        failures.append('fp32_structural_mismatch')
    output_errors = {r['tensor']: r['relative_rmse'] for r in bf16 if r['tensor'] in ('action', 'video')}
    for name, value in output_errors.items():
        if value > output_limit:
            failures.append('bf16_' + name + '_output_mismatch')
    return dict(policy=POLICY, passed=not failures, failures=failures,
                fp32_max_relative_rmse=fp32_max, fp32_limit=fp32_limit,
                bf16_output_relative_rmse=output_errors, bf16_output_limit=output_limit,
                bf16_cache_max_relative_rmse=max((r['relative_rmse'] for r in bf16 if r['tensor'].startswith('layer_')), default=float('inf')),
                bf16_cache_limit_enforced=False, policy_retention_established=False)


@torch.no_grad()
def streamed_fp32(wide, compact, parts, controller, progress=None):
    """Compare the full checkpoint one block pair at a time to bound GPU RAM.

    Both input models must start on CPU. Afterwards all their tensors are on CPU
    with their original dtypes, even if a comparison fails. No weights change.
    """
    if len(wide.blocks) != len(compact.blocks) or len(wide.blocks) != controller.geometry.layers:
        raise ValueError('Different block counts')
    if any(t.device.type != 'cpu' for model in (wide, compact) for t in list(model.parameters()) + list(model.buffers())):
        raise ValueError('Move both video models to CPU before streamed comparison')
    device = parts[1].device
    records = {id(module): {name: t.dtype for name, t in list(module.named_parameters(recurse=False)) + list(module.named_buffers(recurse=False))}
               for model in (wide, compact) for module in model.modules()}
    def restore(module):
        module.cpu()
        for child in module.modules():
            for name, dtype in records[id(child)].items():
                tensor = getattr(child, name)
                if tensor.dtype != dtype:
                    tensor.data = tensor.data.to(dtype=dtype)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    result = []
    cpu_threads = torch.get_num_threads()
    try:
        if device.type == 'cuda':
            torch.set_num_threads(min(8, int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.autocast(device_type=device.type, enabled=False):
            for model in (wide, compact):
                for name, child in model.named_children():
                    if name != 'blocks':
                        child.to(device=device, dtype=torch.float32)
            inputs = {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in parts[0].items()}
            gates = controller.forward_gates(mode='binary')
            indices = torch.tensor(controller.selection()['hidden_indices'], device=device)
            args = (parts[1].float(), parts[2].float(), inputs['context'], inputs['context_mask'], inputs['clip_fea'], inputs['condition_latents'])
            a = wide.pre_dit(*args, hidden_gate=gates[0])
            b = compact.pre_dit(*args)
            result.extend(compare(a[k].index_select(-1, indices), b[k], 'embedding_' + k) for k in ('tokens', 't', 't_mod', 'context'))
            x, y = a['tokens'], b['tokens']
            length = x.shape[1]
            mask = parts[6][:length, :length]
            for i, (left, right) in enumerate(zip(wide.blocks, compact.blocks)):
                left.to(device=device, dtype=torch.float32)
                right.to(device=device, dtype=torch.float32)
                terminal = i == len(wide.blocks) - 1
                f, h = (None, None) if terminal else (gates[1][i], gates[2][i])
                x, k, v = left(x, a['t_mod'], a['freqs'], a['context'], a['context_mask'], mask, gates[0], f, h)
                y, kk, vv = right(y, b['t_mod'], b['freqs'], b['context'], b['context_mask'], mask)
                result.extend([compare(k, kk, f'layer_{i:02d}_k'), compare(v, vv, f'layer_{i:02d}_v'),
                               compare(x.index_select(-1, indices), y, f'layer_{i:02d}_hidden')])
                restore(left)
                restore(right)
                del k, v, kk, vv
                if progress is not None and (i % 10 == 0 or i == len(wide.blocks)-1):
                    progress(dict(layer=i, maximum_relative_rmse=max(r['relative_rmse'] for r in result)))
            av = wide.unpatchify(wide.head(x, a['t'], gates[0]), a['meta']['grid_size'])
            bv = compact.unpatchify(compact.head(y, b['t']), b['meta']['grid_size'])
            result.append(compare(av, bv, 'video'))
        return result
    finally:
        for model in (wide, compact):
            restore(model)
        torch.set_num_threads(cpu_threads)
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
