"""Accept only enumerated, hash-pinned receipts from an exact audited transition.

Staged outside the live repository until a measured optimization passes.
"""
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def verify_performance_receipt(path, receipt, expected):
    path=Path(path).resolve()
    folder=Path(expected['configuration']['root'])/'checks/performance_migrations'
    for migration in sorted(folder.glob('*.json')):
        data=json.loads(migration.read_text())
        if data.get('schema')!='exact-throughput-transition-v1' or not data.get('passed'):
            continue
        if data.get('new_identity')!=expected or data.get('old_identity')!=receipt.get('identity'):
            continue
        old=data['old_identity']
        if old['configuration']!=expected['configuration'] or old['inputs']!=expected['inputs']:
            raise RuntimeError('A throughput transition cannot change the experiment or inputs')
        changed={k for k in set(old['sources'])|set(expected['sources'])
                 if old['sources'].get(k)!=expected['sources'].get(k)}
        if changed!=set(data['changed_sources']):
            raise RuntimeError('Unenumerated source change in throughput transition')
        allowed={'src/vpp2/learned_width_kv/common.py','src/vpp2/learned_width_kv/train.py',
                 'src/vpp2/learned_width_kv/runtime.py','src/vpp2/learned_width_kv/fast_loss.py',
                 'src/vpp2/learned_width_kv/performance.py'}
        if not changed<=allowed:
            raise RuntimeError('Throughput transition changed an unrelated source')
        if data['receipts'].get(str(path))!=sha(path):
            raise RuntimeError('Legacy receipt was not pinned by the throughput audit')
        evidence=[]
        for item in data['evidence']:
            if sha(item['path'])!=item['sha256']:
                raise RuntimeError('Throughput evidence changed')
            evidence.append(json.loads(Path(item['path']).read_text()))
        if len(evidence)!=2:raise RuntimeError('Require paired baseline and candidate evidence')
        base,new=evidence
        if not base['passed'] or not new['passed'] or base['base_identity']!=old or new['base_identity']!=old:
            raise RuntimeError('Throughput evidence does not match the original code')
        if base['checkpoint']!=new['checkpoint'] or base['micro']!=new['micro'] or base['updates']!=new['updates'] or base['updates']<3:
            raise RuntimeError('Profiles were not matched')
        if len(base['rows'])!=base['updates'] or len(new['rows'])!=new['updates']:
            raise RuntimeError('Missing optimizer updates in throughput evidence')
        required={'model','optimizer','controller','mask_optimizer','rng','initial_rng'}
        if len(base['digests'])!=8 or base['digests']!=new['digests'] or any(set(d)!=required for d in base['digests']):
            raise RuntimeError('Model, optimizer, mask or random state changed')
        for a,b in zip(base['rows'],new['rows'],strict=True):
            for key in ('update','sample_sequence_sha256','metrics','mask_gradient_norm','video_gradient_norm'):
                if a[key]!=b[key]:raise RuntimeError('Profile outcomes differ: '+key)
        if new['max_reserved_gib']>72. or new['mean_steady_seconds']>=base['mean_steady_seconds']:
            raise RuntimeError('Candidate has not demonstrated a safe throughput improvement')
        return True
    raise RuntimeError('Receipt identity differs without a matching audited throughput transition: '+str(path))
