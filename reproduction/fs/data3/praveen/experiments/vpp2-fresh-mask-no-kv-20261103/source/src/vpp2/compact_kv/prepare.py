"""Resumable Precision-only preparation using the released native converter."""
import argparse
import csv
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .spec import PRECISION_TASKS, atomic_json, file_sha


def prepare(config):
    import av
    import pyarrow.parquet as pq
    from vpp2.convert_robodojo import load_episodes, episode_table, render_video
    from vpp2.data import DEFAULT_PROMPT, prepare as native_prepare
    root = Path(config["root"])
    media, prepared = root / "data/media", root / "data/prepared"
    media.mkdir(parents=True, exist_ok=True)
    (media / "data").mkdir(exist_ok=True)
    (media / "videos_original").mkdir(exist_ok=True)
    repo = Path(__file__).resolve().parents[3]
    episodes = load_episodes(Path(config["source_dataset"]), repo / "configs/robodojo/source_tasks.csv")
    episodes = [e for e in episodes if e["task"]["task_name"] in PRECISION_TASKS]
    if len(episodes) != 800:
        raise ValueError("Expected 100 original episodes for each of eight Precision tasks")
    rows, previous, table = [], None, None
    for episode in episodes:
        if previous != episode["data"]:
            previous = episode["data"]
            table = pq.read_table(previous, columns=["index", "episode_index", "frame_index", "action", "observation.state"])
        expected = episode_table(table, episode)
        stem = f"episode_{episode['index']:06d}"
        parquet, video = f"data/{stem}.parquet", f"videos_original/{stem}.mp4"
        destination = media / parquet
        if destination.exists():
            if not pq.read_table(destination).equals(expected):
                raise ValueError(f"Existing converted table differs: {destination}")
        else:
            temporary = destination.with_suffix(".partial.parquet")
            pq.write_table(expected, temporary, compression="zstd")
            temporary.replace(destination)
        task = episode["task"]
        rows.append(dict(episode_index=episode["index"], task_name=task["task_name"], dimension="precision",
                         video_path=video, parquet_path=parquet, fps=25, trim_start=0,
                         trim_end=episode["length"], num_frames=episode["length"],
                         high_level_instruction=task["instruction"],
                         training_prompt=DEFAULT_PROMPT.format(task=task["instruction"])))

    def render(pair):
        episode, row = pair
        destination = media / row["video_path"]
        if not destination.exists():
            render_video(episode, destination)
        with av.open(str(destination)) as video:
            stream = video.streams.video[0]
            if (stream.width, stream.height, stream.average_rate, stream.frames) != (880, 480, 25, episode["length"]):
                raise ValueError(f"Invalid native video: {destination}")
        return episode["index"]

    with ThreadPoolExecutor(max_workers=config["convert_workers"]) as pool:
        for i, _ in enumerate(pool.map(render, zip(episodes, rows)), 1):
            if i % 25 == 0:
                print(f"Precision media verified: {i}/800", flush=True)
    metadata = media / "full_episode_metadata.csv"
    with metadata.with_suffix(".partial.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata.with_suffix(".partial.csv").replace(metadata)
    old = Path(config["reference_split"])
    with (old / "val.csv").open() as f:
        heldout = [{"episode_index": int(r["episode_index"]), "task_name": r["task_name"]}
                   for r in csv.DictReader(f) if r["dimension"] == "precision"]
    split = dict(train_episodes=792, val_episodes=8, held_out=heldout)
    if len(heldout) != 8:
        raise ValueError("Reference Precision split differs")
    holdout = root / "data/precision_holdout.json"
    atomic_json(holdout, split)
    stats = Path(config["teacher_bundle"]) / "dataset_stats.json"
    if not (prepared / "preparation.json").exists():
        if prepared.exists():
            raise RuntimeError(f"Partial preparation requires inspection: {prepared}")
        native_prepare(metadata, media, stats, prepared, holdout)
    with (old / "train.csv").open() as f:
        old_rows = list(csv.DictReader(f))
    prompt_sources = {r["training_prompt"]: old / r["text_embedding_path"] for r in old_rows}
    (prepared / "text_embeds").mkdir(exist_ok=True)
    for prompt, relative in json.loads((prepared / "prompts.json").read_text()).items():
        source = prompt_sources[prompt]
        if not source.is_file():
            raise FileNotFoundError(f"Existing native text embedding missing: {source}")
        shutil.copyfile(source, prepared / relative)
    # Compare actual episode identities, not just counts.
    for name in ("train.csv", "val.csv"):
        with (prepared / name).open() as f:
            actual = {int(r["episode_index"]) for r in csv.DictReader(f)}
        with (old / name).open() as f:
            expected = {int(r["episode_index"]) for r in csv.DictReader(f) if r["dimension"] == "precision"}
        if actual != expected:
            raise ValueError(f"Split identities changed: {name}")
    receipt = dict(schema="compact-kv-precision-data-v1", train=792, val=8,
                   teacher_stats_sha256=file_sha(stats),
                   files={str(p.relative_to(prepared)): file_sha(p) for p in sorted(prepared.rglob("*")) if p.is_file()})
    atomic_json(root / "data/ready.json", receipt)
    print(json.dumps({"data_ready": str(root / "data/ready.json"), "train": 792, "val": 8}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    prepare(json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
