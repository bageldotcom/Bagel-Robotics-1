"""Native RGB/label conversion extracted unchanged from the validated converter."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch

UPSTREAM_COMMIT = "72c17acf022494ff63617be86079bb92e92f5bf8"

def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

def require_upstream(root):
    root = Path(root).resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Expected VPP2 revision {UPSTREAM_COMMIT}, found {commit}")
    changes = subprocess.check_output(["git", "diff", "HEAD", "--", "src/vpp2"], cwd=root, text=True)
    if changes.strip():
        raise ValueError("Native VPP2 source has uncommitted changes; use the pinned source")
    sys.path.insert(0, str(root / "src"))
    return root

def camera_frames(path, start, stop, length):
    import av
    count, previous = 0, None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        container.seek(int(start / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Source RGB lacks native timestamps")
            timestamp = float(frame.pts * stream.time_base)
            if timestamp < start - 1e-5:
                continue
            if timestamp >= stop - 1e-5:
                break
            if round((timestamp-start)*25) != count or abs(timestamp-(start+count/25)) > .005 or count >= length:
                raise ValueError("Source RGB timing is not contiguous native 25 Hz")
            previous = frame.to_ndarray(format="rgb24")
            count += 1
            yield previous
    if count == length-1 and previous is not None:
        yield previous
    elif count != length:
        raise ValueError("Only one missing terminal RGB frame may be repeated")

@lru_cache(maxsize=1)
def _read_shard(path):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=["index", "episode_index", "frame_index", "action", "observation.state"],
                         use_threads=False)

def _worker_init(upstream, threads):
    require_upstream(upstream)
    torch.set_num_threads(threads)

def _prepare_episode(task):
    from vpp2.convert_robodojo import episode_table
    from vpp2.policy import compose_training_condition, _normalized_to_uint8
    episode, directory, source_identity, reserve_bytes = task
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    complete = directory / "complete.json"
    if complete.exists():
        marker = json.loads(complete.read_text())
        if marker["source_identity"] != source_identity:
            raise ValueError("Cannot resume an episode from different source data")
        for name, expected in marker["files"].items():
            if file_digest(directory / name) != expected:
                raise ValueError(f"Prepared episode content changed: {directory/name}")
        return marker
    episode_bytes = episode["length"] * (3*240*416 + 2*16*4)
    if shutil.disk_usage(directory).free < reserve_bytes + episode_bytes:
        raise OSError("Insufficient free space for decoded frames plus the configured disk reserve")
    streams = [camera_frames(p, a, b, episode["length"]) for p, a, b in episode["cameras"]]
    frames = np.lib.format.open_memmap(directory / "frames.npy", mode="w+", dtype=np.uint8,
                                       shape=(episode["length"], 3, 240, 416))
    for frame in range(episode["length"]):
        image = compose_training_condition(*(next(stream) for stream in streams)).to(torch.bfloat16)
        frames[frame] = _normalized_to_uint8(image).numpy()
    frames.flush()
    del frames
    for stream in streams:
        if next(stream, None) is not None:
            raise ValueError("Source episode has trailing frames")
    table = episode_table(_read_shard(episode["data"]), episode)
    for output, column in (("actions", "action"), ("states", "observation.state")):
        values = np.asarray(table[column].to_pylist(), dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite demonstration labels/state")
        np.save(directory / f"{output}.npy", values, allow_pickle=False)
    marker = {"source_identity": source_identity,
              "files": {name: file_digest(directory/name) for name in ("frames.npy", "actions.npy", "states.npy")}}
    write_json(complete, marker)
    return marker
