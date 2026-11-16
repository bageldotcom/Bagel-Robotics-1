import random
import numpy as np
import torch
from vpp2.compact_kv.checkpointing import LosslessCheckpointEngine
from vpp2.flexible_mask.preflight import fingerprint


def test_real_compressed_checkpoint_roundtrip(tmp_path):
    import zstandard
    assert zstandard.__version__=="0.25.0"
    engine=LosslessCheckpointEngine()
    value=dict(master=torch.randn(8193,dtype=torch.float32),model=torch.randn(1025,dtype=torch.bfloat16),
        optimizer={"exp_avg":torch.randn(128),"exp_avg_sq":torch.rand(128),"step":torch.tensor(3.)},
        numpy_rng=np.random.get_state(),python_rng=random.getstate(),torch_rng=torch.get_rng_state())
    path=tmp_path/"rank0.pt"
    engine.save(value,path)
    assert path.read_bytes()[:4]==bytes.fromhex("28b52ffd")
    restored=engine.load(path,map_location="cpu")
    assert fingerprint(restored)==fingerprint(value)
    assert list(tmp_path.iterdir())==[path]
    # mmap-backed tensors must still work after the temporary file is unlinked.
    assert torch.equal(restored["master"]+1,value["master"]+1)


def test_real_serializer_accepts_uncompressed_legacy_state(tmp_path):
    path=tmp_path/"legacy.pt";value={"weights":torch.randn(3)}
    torch.save(value,path)
    assert fingerprint(LosslessCheckpointEngine().load(path,map_location="cpu"))==fingerprint(value)
