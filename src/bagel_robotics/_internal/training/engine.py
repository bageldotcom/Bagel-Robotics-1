"""Lossless streaming compression for resumable DeepSpeed optimizer state."""
import os
import shutil
import tempfile
from pathlib import Path
import torch
from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine

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
