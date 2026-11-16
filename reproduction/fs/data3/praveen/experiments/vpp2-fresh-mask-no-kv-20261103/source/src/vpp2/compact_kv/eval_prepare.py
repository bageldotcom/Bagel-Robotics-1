"""Prepare candidate-only native evaluation on node 2 after baseline admission.

Uses the admitted baseline's simulator/transport as a frozen snapshot. Never
submits or modifies an original-model evaluation.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .spec import PRECISION_TASKS, atomic_json, file_sha


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def prepare(args):
    baseline, destination, export, source = map(Path, (args.baseline, args.destination, args.export, args.source))
    # Read-only admission: the baseline workflow owns fixes to its simulator.
    sys.path.insert(0,str(baseline))
    from native_eval.prepare import verify
    from native_eval.diagnostics import admit_full
    baseline_id=verify()
    admit_full('vpp2')
    del sys.path[0]
    ready=json.loads((export/'COMPLETE.json').read_text())
    reload=json.loads((export/'RELOAD_CHECK.json').read_text())
    if ready['update']!=1000 or ready['candidate']!=args.candidate or not reload['passed']:
        raise RuntimeError('Candidate needs an immutable, reload-verified update-1000 export')
    for name,sha in ready['files'].items():
        if file_sha(export/name)!=sha:raise RuntimeError(f'Changed export artifact: {name}')
    master=json.loads((baseline/'inputs/episodes.json').read_text())['episodes']
    selected=[]
    for task in PRECISION_TASKS:
        choices=[r for r in master if r['suite']=='precision' and r['task']==task]
        if len(choices)!=50:raise RuntimeError('Expected 50 available baseline identities per task')
        selected.extend(choices[:10])
    if len(selected)!=80 or len({e['id'] for e in selected})!=80:
        raise RuntimeError('Need 80 unique Precision pilot episodes')
    if destination.exists():
        raise FileExistsError('Evaluation is immutable; reuse its worker to resume, not prepare again')
    destination.mkdir(parents=True)
    (destination/'inputs').mkdir()
    (destination/'source').mkdir()
    # Copy source code, sharing only immutable heavyweight assets.
    shutil.copytree(baseline/'native_eval', destination/'native_eval',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(baseline/'source/simulator',destination/'source/simulator',symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__','.git','.cache','Assets'))
    (destination/'source/simulator/Assets').symlink_to((baseline/'source/simulator/Assets').resolve(),target_is_directory=True)
    shutil.copytree(baseline/'source/transport_deps',destination/'source/transport_deps',symlinks=True,
                   ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(source/'src',destination/'source/vpp2/src',ignore=shutil.ignore_patterns('__pycache__'))
    policy=json.loads((baseline/'inputs/vpp2-policy.json').read_text())
    policy.update(checkpoint_path=str(export/'action.pt'),video_checkpoint_path=str(export/'video.pt'),
                  dataset_stats_path=str(export/'dataset_stats.json'),ckpt_name=f'compact-{args.candidate}-1000')
    atomic_json(destination/'inputs/vpp2-policy.json',policy)
    atomic_json(destination/'inputs/episodes.json',dict(episodes=selected,baseline_manifest_sha256=file_sha(baseline/'inputs/episodes.json')))
    # Source wrappers only change manifest selection/verification and scheduling.
    # Native model, transport, simulator, observation processing and action path stay unchanged.
    templates=source/'evaluation/templates'
    for name in ('prepare.py','scope.py','candidate_worker.py'):
        shutil.copyfile(templates/name,destination/'native_eval'/name)
    shutil.copyfile(templates/'candidate_python.py',destination/'candidate_python.py')
    os.chmod(destination/'candidate_python.py',0o755)
    sources={str(p.relative_to(destination)):file_sha(p) for folder in ('native_eval','source')
             for p in sorted((destination/folder).rglob('*'))
             if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and 'Assets' not in p.parts}
    sources['candidate_python.py']=file_sha(destination/'candidate_python.py')
    protocol=dict(version='compact-kv-precision-native-v1',candidate=args.candidate,update=1000,
        baseline_protocol=baseline_id,baseline_root=str(baseline),source_hashes=sources,
        inputs={name:file_sha(destination/'inputs'/name) for name in ('episodes.json','vpp2-policy.json')},
        export=str(export),export_receipt_sha256=file_sha(export/'COMPLETE.json'),
        order=['vpp2','internw0'],active_policies=['vpp2'],scope=True,
        gpu_sharing=dict(mode='shared-shard',workers_per_gpu=1),
        primary_episodes=80,repeat_episodes=8,repeat_ids=[next(e['id'] for e in selected if e['task']==task) for task in PRECISION_TASKS],
        native_inference=dict(seed=1,steps=10,shift=1,horizon=32,replan=24),
        instruction='Compare paired original identities; no original policy evaluation is scheduled here.')
    atomic_json(destination/'protocol.json',dict(id=digest(protocol),**protocol))
    print(json.dumps(dict(destination=str(destination),candidate=args.candidate,primary=80,repeats=8)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--export',required=True);p.add_argument('--source',required=True)
    p.add_argument('--candidate',choices=['E1','E2'],required=True)
    prepare(p.parse_args())

if __name__=='__main__':main()
