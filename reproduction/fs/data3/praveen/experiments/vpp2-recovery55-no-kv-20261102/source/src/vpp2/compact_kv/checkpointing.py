"""Atomic native exports and lossless compressed DeepSpeed resumable state."""
import io
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch
import torch.distributed as dist
from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine

from .spec import architecture, atomic_json, file_sha


class LosslessCheckpointEngine(TorchCheckpointEngine):
    """Stream Zstandard, preserving every bit of optimizer/master weights.

    Files retain DeepSpeed's expected .pt names. This engine, not plain torch.load,
    must read them. Native exports below remain ordinary uncompressed .pt files.
    """
    def save(self, state_dict, path):
        import zstandard as zstd
        path = Path(path)
        temporary = path.with_name(path.name + '.partial')
        with temporary.open('wb') as raw:
            with zstd.ZstdCompressor(level=1, threads=1).stream_writer(raw, closefd=False) as stream:
                torch.save(state_dict, stream)
            raw.flush()
            os.fsync(raw.fileno())
        temporary.replace(path)

    def load(self, path, map_location=None):
        import zstandard as zstd
        with open(path, 'rb') as raw:
            magic = raw.read(4)
            raw.seek(0)
            if magic != b'\x28\xb5\x2f\xfd':
                return torch.load(raw, map_location=map_location, weights_only=False)
            # A seekable BytesIO would hold the complete decompressed optimizer
            # plus a second copy while torch.load rebuilds it on every rank. Use
            # local checkpoint storage and mmap to keep resume RAM bounded.
            fd,name=tempfile.mkstemp(prefix='.'+Path(path).name+'-load-',suffix='.pt',dir=os.environ.get('VPP2_CHECKPOINT_LOAD_TMPDIR',str(Path(path).parent)))
            try:
                with os.fdopen(fd,'wb') as contents, zstd.ZstdDecompressor().stream_reader(raw) as stream:
                    shutil.copyfileobj(stream,contents,length=8*1024*1024)
                return torch.load(name,map_location=map_location,weights_only=False,mmap=True)
            finally:
                # Linux mappings keep their storage alive after unlink. No old
                # checkpoint or unrelated file is removed by this local cleanup.
                os.unlink(name)


def rng_state():
    return dict(torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state(),
                python=random.getstate(), numpy=np.random.get_state())


def restore_rng(state):
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state(state['torch_cuda'])
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])


def save_training(engine, directory, config, candidate, update, provenance):
    rank, world = dist.get_rank(), dist.get_world_size()
    states = [None] * world
    dist.all_gather_object(states, rng_state())
    root = Path(directory) / 'checkpoints'
    tag = f'update_{update:06d}'
    client = dict(update=update, next_global_sample=update * config['global_batch'], config=config,
                  candidate=candidate, provenance=provenance, rank_rng=states,
                  serialization='lossless-zstd-torch-v1')
    engine.save_checkpoint(str(root), tag=tag, client_state=client)
    dist.barrier()
    if rank == 0:
        files = list((root / tag).rglob('*.pt'))
        if not files or any(p.stat().st_size == 0 for p in files):
            raise RuntimeError('Missing resumable checkpoint partitions')
        atomic_json(root / tag / 'COMPLETE.json', dict(update=update, candidate=candidate,
                files={str(p.relative_to(root / tag)): p.stat().st_size for p in files}))
        # Retain the previous checkpoint until the replacement is fully committed.
        for old in root.glob('update_*'):
            if old.is_dir() and old.name != tag and (old / 'COMPLETE.json').is_file():
                shutil.rmtree(old)
    dist.barrier()


def export_native(engine, directory, config, candidate, update, provenance):
    rank = dist.get_rank()
    root = Path(directory) / 'exports'
    destination = root / f'update_{update:06d}'
    temporary = root / f'.update_{update:06d}.partial'
    state = engine._zero3_consolidated_16bit_state_dict(exclude_frozen_parameters=False)
    if rank == 0:
        if destination.exists():
            receipt = json.loads((destination / 'COMPLETE.json').read_text())
            if receipt['provenance'] != provenance:
                raise RuntimeError('Refusing to replace an export with different provenance')
        else:
            temporary.mkdir(parents=True, exist_ok=True)
            video_state = {k.removeprefix('video.'): v for k, v in state.items()}
            if any(not k.startswith('video.') for k in state):
                raise RuntimeError('Unexpected non-video parameter in compact student export')
            torch.save(dict(dit=video_state, step=update, origin_step=100000,
                            compact_architecture=provenance['architecture']), temporary / 'video.pt')
            source = Path(config['teacher_bundle'])
            action = torch.load(source / 'action.pt', map_location='cpu', weights_only=True, mmap=True)
            action['origin_step'], action['step'] = int(action['step']), update
            action['video_checkpoint'] = {'path': 'video.pt'}
            if candidate in ('W1', 'W2'):
                from vpp2.checkpoint_compat import canonicalize_config
                action['resolved_config'] = canonicalize_config(action['resolved_config'])
                action['resolved_config']['model']['video_dit_config'].update(
                    dim=2048, hidden_dim=2048, ffn_dim=8192, num_layers=40, num_heads=40, attn_head_dim=128)
            torch.save(action, temporary / 'action.pt')
            shutil.copyfile(source / 'dataset_stats.json', temporary / 'dataset_stats.json')
            atomic_json(temporary / 'compact_architecture.json', provenance['architecture'])
            # Native loader uses the adjacent architecture descriptor and no teacher.
            atomic_json(temporary / 'COMPLETE.json', dict(candidate=candidate, update=update,
                provenance=provenance, inference_teacher_required=False,
                files={p.name: file_sha(p) for p in temporary.iterdir() if p.is_file()}))
            temporary.replace(destination)
    dist.barrier()
    return destination
