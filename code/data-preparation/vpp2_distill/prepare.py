"""Prepare lossless native frames and every EE16 label using the upstream data reader."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import multiprocessing
import time
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from .contracts import DATA_FORMAT, PREPROCESSING, UPSTREAM_COMMIT, EE16Normalizer, digest, file_digest, native_text_context, write_json


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


def prepare(args):
    upstream = require_upstream(args.vpp2_root)
    from vpp2.convert_robodojo import load_episodes
    root, source = Path(args.output).resolve(), Path(args.source).resolve()
    prepared = Path(args.prepared).resolve()
    workers, threads = getattr(args, "workers", 1), getattr(args, "threads", 2)
    reserve_bytes = int(getattr(args, "min_free_gib", 256) * 2**30)
    if workers < 1 or threads < 1 or reserve_bytes < 0:
        raise ValueError("workers/threads must be positive and disk reserve non-negative")
    rows = {}
    for split, name in (("train", "train.csv"), ("validation", "val.csv")):
        with (prepared / name).open() as handle:
            for row in csv.DictReader(handle):
                index = int(row["episode_index"])
                if index in rows:
                    raise ValueError("Training and validation episodes overlap")
                rows[index] = (split, row)
    episodes = load_episodes(source, upstream / "configs/robodojo/source_tasks.csv")
    if set(rows) != {e["index"] for e in episodes}:
        raise ValueError("Prepared split must cover the complete native dataset")
    normalizer = EE16Normalizer(prepared / "dataset_stats.json")
    identity = dict(format=DATA_FORMAT, preprocessing=PREPROCESSING, stats_sha256=normalizer.identity,
                    source_info_sha256=file_digest(source / "meta/info.json"),
                    split_sha256={n: file_digest(prepared / n) for n in ("train.csv", "val.csv")})
    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / "preparation_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Output directory contains another preparation identity")
    write_json(identity_path, identity)
    shutil.copyfile(prepared / "dataset_stats.json", root / "dataset_stats.json")
    output_rows, tasks, texts = [], [], {}
    for episode in episodes:
        split, row = rows[episode["index"]]
        # These are full native episodes: no target shift or temporal subsampling.
        if abs(float(row["trim_start"])) > 1e-6 or abs(float(row["trim_end"])-episode["length"]) > 1e-6:
            raise ValueError("Expected untrimmed native episodes in the reproduced split")
        relative = f"episodes/{episode['index']:05d}"
        text_path = Path(row["text_embedding_path"])
        if not text_path.is_absolute():
            text_path = prepared / text_path
        text_key = digest(row["training_prompt"])
        if text_key not in texts:
            text_out = root / "text" / f"{text_key}.pt"
            text_out.parent.mkdir(exist_ok=True)
            text = torch.load(text_path, map_location="cpu", weights_only=True)
            if tuple(text["context"].shape) != (128, 4096) or tuple(text["mask"].shape) != (128,):
                raise ValueError("Expected native UMT5 context length 128")
            text["context"], text["mask"] = native_text_context(text["context"], text["mask"])
            torch.save({**text, "prompt": row["training_prompt"]}, text_out)
            texts[text_key] = (text_out.relative_to(root).as_posix(), file_digest(text_out), file_digest(text_path))
        text_relative, text_hash, source_text_hash = texts[text_key]
        source_identity = digest({"preparation": identity, "episode": episode["index"], "length": episode["length"],
                                  "text": source_text_hash})
        tasks.append((episode, root / relative, source_identity, reserve_bytes))
        output_rows.append(dict(id=str(episode["index"]), length=episode["length"], path=relative,
                                task=row["task_name"], prompt=row["training_prompt"], split=split,
                                text=text_relative, text_sha256=text_hash))
    needed = sum(e["length"] * (3*240*416 + 2*16*4) for e,d,_,_ in tasks if not (d/"complete.json").exists())
    if shutil.disk_usage(root).free < needed + reserve_bytes:
        raise OSError("Insufficient space for the remaining decoded dataset plus disk reserve")
    started, frames_done = time.monotonic(), 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_worker_init, initargs=(str(upstream), threads)) as pool:
        for i, marker in enumerate(pool.map(_prepare_episode, tasks, chunksize=1)):
            output_rows[i]["files"] = marker["files"]
            frames_done += output_rows[i]["length"]
            elapsed = time.monotonic()-started
            print(json.dumps(dict(episodes_done=i+1, episodes_total=len(episodes), frames_done=frames_done,
                                  seconds=round(elapsed, 1), frames_per_second=round(frames_done/elapsed, 2))), flush=True)
    manifest = dict(identity, episodes=output_rows, upstream_commit=UPSTREAM_COMMIT)
    write_json(root / "manifest.json", {**manifest, "sha256": digest(manifest)})
    print(json.dumps(dict(passed=True, episodes=len(episodes), frames=frames_done, manifest=digest(manifest))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vpp2-root", required=True)
    parser.add_argument("--source", required=True, help="Official LeRobot v3 dataset root")
    parser.add_argument("--prepared", required=True, help="Existing VPP2 train.csv/val.csv/stats/text directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2, help="Torch CPU threads per worker")
    parser.add_argument("--min-free-gib", type=float, default=256)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    prepare(args)


if __name__ == "__main__":
    main()

