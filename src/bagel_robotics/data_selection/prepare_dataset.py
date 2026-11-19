#!/usr/bin/env python3
"""Convert public RoboDojo EE16 data to the recorded native training format.

A sample writes only a sample receipt. Only a complete hash-matching conversion
publishes the training manifest. Existing incompatible outputs are never replaced.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import sys

from ..configuration import resource, read_json
import gzip
import urllib.request
ROOT = Path(__file__).resolve().parents[1]

def native_manifest():
    return json.loads(gzip.decompress(resource("native_manifest.json.gz").read_bytes()))

def conditioning(output, row):
    path = safe_child(output, row["text"])
    if path.exists():
        if sha256(path) != row["text_sha256"]:
            raise ValueError("Text conditioning checksum differs")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    revision = read_json(resource("data.json"))["conditioning_revision"]
    url = "https://raw.githubusercontent.com/bageldotcom/Bagel-Robotics-1/" + revision + "/data/" + row["text"]
    temporary = path.with_suffix(".partial")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    if sha256(temporary) != row["text_sha256"]:
        temporary.unlink()
        raise ValueError("Downloaded text conditioning checksum differs")
    temporary.replace(path)
    return path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def safe_child(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root):
        raise ValueError('Dataset path escapes its root: ' + relative)
    return path


def verify_episode(root, row):
    errors = []
    folder = safe_child(root, row['path'])
    for name, expected in row['files'].items():
        path = safe_child(folder, name)
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"episode {row['id']}: {name} is missing or differs")
    return errors


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--source', help='Public EE16 LeRobot v3 dataset directory')
    p.add_argument('--vpp2-root', help='Clean data-reader checkout at the pinned commit')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--min-free-gib', type=float, default=256)
    p.add_argument('--episodes', type=int, nargs='+', help='Sample conversion only')
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if a.workers < 1 or a.threads < 1 or a.min_free_gib < 0:
        p.error('workers/threads must be positive; disk reserve must be nonnegative')
    manifest_bytes = gzip.decompress(resource('native_manifest.json.gz').read_bytes())
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    rows = manifest['episodes']
    ids = {int(r['id']) for r in rows}
    if a.episodes is not None:
        if len(a.episodes) != len(set(a.episodes)) or not set(a.episodes) <= ids:
            p.error('Sample IDs must be unique and present in the manifest')
        rows = [r for r in rows if int(r['id']) in a.episodes]
    out = Path(a.output).expanduser().resolve()
    if out == ROOT or out.is_relative_to(ROOT):
        p.error('Use a separate output directory')
    if a.dry_run:
        print(json.dumps(dict(episodes=len(rows), full_dataset=a.episodes is None,
            output=str(out), decoded_bytes=sum(r['length']*(3*240*416+128) for r in rows),
            conversion='native RGB, BF16 round-trip, 25 Hz, absolute EE16',
            manifest_sha256=manifest_hash), indent=2))
        return 0
    if not a.verify_only:
        if not a.source or not a.vpp2_root:
            p.error('Conversion requires --source and --vpp2-root')
        from .conversion import require_upstream, _prepare_episode, _worker_init
        require_upstream(a.vpp2_root)
        from vpp2.convert_robodojo import load_episodes
        source = Path(a.source).resolve()
        if sha256(source/'meta/info.json') != manifest['source_info_sha256']:
            raise ValueError('Source metadata does not match the pinned dataset')
        episodes = {e['index']: e for e in load_episodes(source,
                    Path(a.vpp2_root)/'configs/robodojo/source_tasks.csv')}
        out.mkdir(parents=True, exist_ok=True)
        needed = sum(r['length']*(3*240*416+128) for r in rows
                     if not (safe_child(out,r['path'])/'complete.json').exists())
        if shutil.disk_usage(out).free < needed + int(a.min_free_gib*2**30):
            raise OSError('Not enough storage for conversion and the requested reserve')
        tasks = []
        manifest_hash=hashlib.sha256(manifest_bytes).hexdigest()
        for row in rows:
            e = episodes[int(row['id'])]
            if e['length'] != row['length'] or e['task']['task_name'] != row['task']:
                raise ValueError('Episode identity differs from the recorded data')
            identity=hashlib.sha256(json.dumps({'manifest':manifest_hash,
                'episode':row['id']},sort_keys=True).encode()).hexdigest()
            tasks.append((e,safe_child(out,row['path']),identity,int(a.min_free_gib*2**30)))
        with ProcessPoolExecutor(max_workers=a.workers,
                mp_context=multiprocessing.get_context('spawn'),initializer=_worker_init,
                initargs=(str(Path(a.vpp2_root).resolve()),a.threads)) as pool:
            for row, marker in zip(rows,pool.map(_prepare_episode,tasks,chunksize=1)):
                if marker['files'] != row['files']:
                    raise ValueError('Native conversion differs at episode '+row['id'])
                print('Verified episode '+row['id'],flush=True)
    errors=[e for row in rows for e in verify_episode(out,row)]
    if errors:raise ValueError('\n'.join(errors[:30]))
    out.mkdir(parents=True,exist_ok=True)
    for row in rows:
        dest=safe_child(out,row['text']);dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and sha256(dest)!=row['text_sha256']:
            raise ValueError('Refusing to replace different text conditioning')
        if not dest.exists():conditioning(out,row)
    stats=out/'dataset_stats.json'
    if stats.exists() and sha256(stats)!=sha256(resource('normalization.json')):
        raise ValueError('Refusing to replace different normalization')
    if not stats.exists():shutil.copy2(resource('normalization.json'),stats)
    full=a.episodes is None
    receipt=dict(passed=True,episodes=len(rows),full_dataset=full,
        manifest_sha256=manifest_hash,all_episode_arrays_sha256_verified=True)
    if full:
        dest=out/'manifest.json'
        if dest.exists() and sha256(dest)!=manifest_hash:
            raise ValueError('Output has another dataset manifest')
        dest.write_bytes(manifest_bytes)
    (out/('verification.json' if full else 'sample-verification.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
