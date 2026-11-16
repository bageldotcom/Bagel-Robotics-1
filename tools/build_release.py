"""Maintainer-only: assemble a weight-free release from the verified local runs.

This does not launch jobs, modify original experiments, or publish anything.
"""
from pathlib import Path
import hashlib
import json
import shutil

DEST = Path(__file__).resolve().parents[1]
EXP = Path('/data3/praveen/experiments')
RUNS = {
    'mask_original': ('vpp2-flexible-mask-recovery-20261031', 'source-608bc008a845'),
    'mask_fast': ('vpp2-flexible-mask-recovery-20261031-speed4-micro4', 'source-6d0fd9ee9e16'),
    'mask_final': ('vpp2-flexible-mask-pressure2x-20261101', 'source'),
    'compact_parent': ('vpp2-selected-mask-recovery-20261101', 'source'),
    'recovery_pilot': ('vpp2-recovery55-no-kv-20261102', 'source'),
    'recovery_1000': ('vpp2-recovery55-no-kv-to1000-20261103', 'source'),
}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def put(p):
    """Keep absolute paths as data, under a contained filesystem image."""
    p = Path(p)
    q = DEST / 'reproduction' / 'fs' / p.relative_to('/')
    if q.exists():
        if sha(p) != sha(q):
            raise ValueError('Conflicting archived file: '+str(p))
        return
    q.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, q)

def metadata(root):
    root = Path(root)
    if not root.exists():
        return
    for name in ('config.json', 'inputs.json', 'source_manifest.json', 'selected.json',
                 'export_verified.json', 'export_pending_verification.json'):
        p = root/name
        if p.is_file(): put(p)
    for folder in ('checks', 'control'):
        for p in (root/folder).glob('*'):
            if p.is_file() and p.suffix in ('.json', '.py', '.txt', '.md') and p.stat().st_size < 10_000_000:
                if p.name not in ('stop_request.json', 'completed.json', 'handoff_state.json'):
                    put(p)
    for p in (root/'slurm').glob('*.sbatch'):
        put(p)

def main():
    records = {}
    for key, (name, srcname) in RUNS.items():
        root = EXP/name
        metadata(root)
        manifest = json.loads((root/'source_manifest.json').read_text())
        src = root/srcname
        for name, expected in manifest['files'].items():
            p = src/name
            if sha(p) != expected: raise ValueError('Source hash mismatch: '+str(p))
            put(p)
        for p in src.rglob('requirements*.txt'):
            put(p)
        recorded_identity=manifest.get('identity') or json.loads((root/'inputs.json').read_text())['identity']
        records[key] = dict(root=str(root), source=str(src), identity=recorded_identity,
                            source_files=len(manifest['files']))
    # Small records required by the archived provenance checks. No tensor files.
    roots = [
        EXP/'vpp2-flexible-mask-recovery-20261031-micro2',
        EXP/'vpp2-selected-mask-recovery-curated30k-20261101',
        EXP/'vpp2-mask-checkpoint-selection-20261101/node3',
    ]
    for r in roots: metadata(r)
    pending=[EXP/name for name,_ in RUNS.values()];visited=set()
    while pending:
        r=pending.pop()
        if str(r) in visited:continue
        visited.add(str(r));metadata(r)
        c=json.loads((r/'config.json').read_text())
        parent=c.get('resume_from')
        if parent:
            pr=Path(parent['root']);pending.append(pr)
            receipt=pr/'mask/checkpoints'/parent['tag']/'COMPLETE.json'
            if receipt.exists():put(receipt)
    config = json.loads((EXP/RUNS['recovery_pilot'][0]/'config.json').read_text())
    for entry in config['artifacts'].values(): put(entry['path'])
    put(config['panel_file'])
    put(Path(config['curation_artifacts'])/'inventory.json')
    for r in [EXP/'vpp2-selected-mask-recovery-curated30k-20261101',
              EXP/'vpp2-selected-mask-recovery-20261101']:
        for step in [55, 250]:
            p = r/f'recovery/diagnostics/native_{step:06d}.json'
            if p.exists(): put(p)
    sel = EXP/'vpp2-mask-checkpoint-selection-20261101'
    for p in (sel/'node3').rglob('*.json'):
        if p.stat().st_size < 20_000_000 and 'diagnostics' not in p.parts:
            put(p)
    for root in [EXP/RUNS['mask_fast'][0], EXP/RUNS['mask_final'][0]]:
        for p in (root/'mask/checkpoints').glob('update_*/COMPLETE.json'): put(p)
    # Receipts are useful without bundling hundreds of GB of optimizer tensors.
    for root in [EXP/RUNS['compact_parent'][0], EXP/RUNS['recovery_pilot'][0], EXP/RUNS['recovery_1000'][0]]:
        for p in (root/'recovery/checkpoints').glob('update_*/COMPLETE.json'): put(p)
    # Source evidence for results; these contain numbers and sample identities only.
    out = DEST/'results/evidence'; out.mkdir(parents=True,exist_ok=True)
    for key in ('mask_fast','mask_final','recovery_pilot','recovery_1000'):
        root = EXP/RUNS[key][0]
        stage = 'mask' if key.startswith('mask') else 'recovery'
        target = out/key; target.mkdir(exist_ok=True)
        for p in (root/stage/'diagnostics').glob('native_*.json'):
            shutil.copyfile(p,target/p.name)
        for p in (root/stage/'diagnostics').glob('selection_*.json'):
            shutil.copyfile(p,target/p.name)
    baseline=EXP/'vpp2-selected-mask-recovery-curated30k-20261101/recovery/diagnostics/native_000055.json'
    if not (out/'recovery_pilot/native_000055.json').exists():
        shutil.copyfile(baseline,out/'recovery_pilot/native_000055.json')
    best=EXP/RUNS['recovery_1000'][0]/'recovery/best_models.json'
    shutil.copyfile(best,out/'recovery_best_models.json')
    license_file=Path('/data1/praveen/experiments/vpp2-joint-feature-cache-20261013/source/LICENSE')
    (DEST/'LICENSES').mkdir(exist_ok=True)
    shutil.copyfile(license_file,DEST/'LICENSES/VPP2-upstream-MIT.txt')
    (DEST/'reproduction/runs.json').write_text(json.dumps(records,indent=2)+'\n')
    files={str(p.relative_to(DEST)):dict(sha256=sha(p),bytes=p.stat().st_size)
           for folder in ('reproduction','results/evidence','LICENSES')
           for p in sorted((DEST/folder).rglob('*')) if p.is_file() and p != DEST/'reproduction/release_files.json'}
    (DEST/'reproduction/release_files.json').write_text(json.dumps(files,indent=2)+'\n')
    print(json.dumps(dict(files=len(files),bytes=sum(v['bytes'] for v in files.values())),indent=2))

if __name__=='__main__':main()
