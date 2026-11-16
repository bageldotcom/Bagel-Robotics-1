"""Real-checkpoint correctness gates; no policy performance threshold."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from hydra.utils import instantiate
from torch.utils.data import default_collate

from .blocks import install_blocks, KVOnlyBlock, CacheFullBlock, TerminalCacheBlock
from .model import native_config, load_reference, prepare_batch, native_cache, action_prediction, infer_from_cache
from .sampling import PositionDataset
from .spec import atomic_json, file_sha


def tensor_hash(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[3]
    paths = ['src/vpp2/compact_kv/model.py', 'src/vpp2/compact_kv/blocks.py',
             'src/vpp2/compact_kv/losses.py', 'src/vpp2/models/wan21_14b/mot.py',
             'src/vpp2/models/wan21_14b/helpers/loader.py']
    return {p: file_sha(root / p) for p in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config['root'])
    if not (root / 'data/ready.json').is_file():
        raise RuntimeError('Native data must pass preparation first')
    torch.cuda.set_device(0)
    torch.manual_seed(config['seed'])
    dataset = PositionDataset(instantiate(native_config(config).data.train), config['seed'])
    sample = default_collate([dataset[0]])
    model = load_reference(config)
    print('Reference model and first Precision window loaded', flush=True)
    action_hash, proprio_hash = tensor_hash(model.action_expert), tensor_hash(model.proprio_encoder)
    prepared = prepare_batch(model, sample, config['seed'])
    inputs, noise, t_v, noisy_action, t_a, target, mask = prepared
    with torch.no_grad():
        baseline = native_cache(model, inputs, noise, t_v, mask)
        if len(baseline) != 40 or any(x[k].shape != (1, 1950, 5120) for x in baseline for k in ('k', 'v')):
            raise ValueError('Native cache is not the required 40 x K,V interface')
        pred0 = action_prediction(model, inputs, baseline, noisy_action, t_a, mask)
        chunk0 = infer_from_cache(model, inputs, baseline, mask, [0], config['seed'])
        # Install all full blocks: no parameters changed, only refactoring the forward.
        install_blocks(model.video_expert, ())
        wrapped = native_cache(model, inputs, noise, t_v, mask)
        maximum = 0.
        for i, (a, b) in enumerate(zip(baseline, wrapped)):
            for key in ('k', 'v'):
                maximum = max(maximum, float((a[key] - b[key]).abs().max()))
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=f'Full wrapper {i}/{key}')
        pred1 = action_prediction(model, inputs, wrapped, noisy_action, t_a, mask)
        torch.testing.assert_close(pred0, pred1, rtol=0, atol=0)
        del wrapped
        model.video_expert.blocks[39].__class__ = TerminalCacheBlock
        terminal = native_cache(model, inputs, noise, t_v, joint_mask=mask)
        for i, (a, b) in enumerate(zip(baseline, terminal)):
            for key in ('k', 'v'):
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=f'Terminal execution {i}/{key}')
        terminal_chunk = infer_from_cache(model, inputs, terminal, mask, [0], config['seed'])
        torch.testing.assert_close(chunk0, terminal_chunk, rtol=0, atol=0)
        del terminal, terminal_chunk
        model.video_expert.blocks[39].__class__ = CacheFullBlock
        model.video_expert.blocks[39] = KVOnlyBlock(model.video_expert.blocks[39])
        final_only = native_cache(model, inputs, noise, t_v, mask)
        for i, (a, b) in enumerate(zip(baseline, final_only)):
            for key in ('k', 'v'):
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=f'Last-slot parity {i}/{key}')
        chunk1 = infer_from_cache(model, inputs, final_only, mask, [0], config['seed'])
        torch.testing.assert_close(chunk0, chunk1, rtol=0, atol=0)
        del final_only
    leaves = [{k: value.detach().requires_grad_() for k, value in layer.items()} for layer in baseline]
    # Enable only native MoT checkpointing; frozen weights stay frozen.
    model.mot.training = True
    pred = action_prediction(model, inputs, leaves, noisy_action, t_a, mask)
    (pred.float() - target.float()).square().mean().backward()
    gradient_norms = [[float(layer[k].grad.float().norm()) for k in ('k', 'v')] for layer in leaves]
    if not all(x > 0 and x < float('inf') for row in gradient_norms for x in row):
        raise RuntimeError('A frozen-action backward failed to reach some video K/V slots')
    if any(p.grad is not None or p.requires_grad for p in model.parameters()):
        raise RuntimeError('Frozen model has parameter gradients or trainable parameters')
    assert action_hash == tensor_hash(model.action_expert)
    assert proprio_hash == tensor_hash(model.proprio_encoder)
    atomic_json(root / 'checks/native_gpu.json', dict(passed=True, time=time.time(),
            all_40_full_block_kv_max_absolute_difference=maximum, final_kv_only_10_step_actions_exact=True,
            terminal_cache_all_40_kv_exact=True, terminal_cache_10_step_actions_exact=True,
            action_gradients_to_all_40_kv_slots=gradient_norms, action_sha256=action_hash,
            proprio_sha256=proprio_hash, source_hashes=source_hashes(),
            max_gpu_memory_bytes=torch.cuda.max_memory_allocated(), torch_version=torch.__version__))
    print('PASS: native wrapper parity, final K/V slot parity, native 10-step action parity, 80 cache gradients, frozen action/proprio', flush=True)


if __name__ == '__main__':
    main()
