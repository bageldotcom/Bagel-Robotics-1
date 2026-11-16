"""Evidence-bound readiness. CPU checks never grant GPU training readiness."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch
from .config import load,identity
from vpp2.full_mask.panel import file_hash
from vpp2.compact_kv.spec import atomic_json


def require_receipt(config,name):
    root=Path(config['root']);p=root/'checks'/f'{name}.json'
    value=json.loads(p.read_text())
    if value.get('passed') is not True or value.get('identity')!=identity(config):
        raise RuntimeError('Missing/stale verification: '+str(p))
    return value


def fingerprint(value):
    import numpy as np
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            y=x.detach().cpu().contiguous();h.update(str((y.dtype,tuple(y.shape))).encode());h.update(y.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            for item in x:visit(item)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def frozen_fingerprint(runtime):
    return fingerprint({n:getattr(p,'ds_tensor',p) for n,p in runtime.reference.named_parameters()})


def runtime_fingerprint(runtime):
    from .rng import rng_state
    return {name:fingerprint(value) for name,value in dict(
        video={n:getattr(p,'ds_tensor',p) for n,p in runtime.engine.module.named_parameters()},
        optimizer=runtime.engine.optimizer.state_dict(),
        masks=None if runtime.masks is None else runtime.masks.state_dict(),
        mask_optimizer=None if runtime.mask_optimizer is None else runtime.mask_optimizer.state_dict(),
        rng=rng_state()).items()}


def gpu_contracts(runtime,panel):
    from vpp2.adaptive_mask.data import prepare
    from .rng import diagnostic_rng
    from vpp2.adaptive_mask.equivalence import compare
    from .runtime import independent_storage
    row=next(r for r in panel['windows'] if r['role']=='selection_initial')
    reports=[]
    with diagnostic_rng(),torch.no_grad():
        parts=prepare(runtime.reference,runtime.reader,[row],[0],runtime.config['seed'],[0])
        teacher=runtime.video(parts,teacher=True)
        # Open, alpha-zero and hard-all-one paths are distinct contracts.
        student=runtime.video(parts,gates=runtime.gates(0))
        reports.append(compare(student['video'],teacher['video'],'video'))
        reports.append(compare(runtime.predict(parts,student['cache']),runtime.predict(parts,teacher['cache']),'action'))
        for layer,(a,b) in enumerate(zip(student['cache'],teacher['cache'],strict=True)):
            for k in ('k','v'):
                if a[k].shape!=(1,1950,5120) or b[k].requires_grad:raise ValueError('Native cache contract failed')
                reports.append(compare(a[k],b[k],f'layer_{layer:02d}_{k}'))
        maximum=max(r['relative_rmse'] for r in reports)
        if maximum>1e-6:raise RuntimeError('All-open student does not reproduce teacher: '+str(maximum))
    independent_storage(runtime.engine.module,runtime.teacher.module)
    runtime.check_frozen()
    return dict(passed=True,all_open_max_relative_rmse=maximum,all_40_cache_shapes=True,
                separate_student_storage=True,reports=reports)


def cpu(config):
    root=Path(config['root']);repo=Path(__file__).resolve().parents[3]
    inputs=json.loads((root/'inputs.json').read_text())
    if inputs['identity']!=identity(config):raise ValueError('Prepared inputs/source identity changed')
    output=root/'checks/pytest.txt'
    with output.open('w') as f:
        result=subprocess.run([sys.executable,'-m','pytest','tests/flexible_mask','tests/adaptive_mask/test_equivalence.py',
            'tests/adaptive_mask/test_physical.py','tests/adaptive_mask/test_execution.py',
            '-q'],cwd=repo,stdout=f,stderr=subprocess.STDOUT)
    report=dict(identity=identity(config),passed=result.returncode==0,pytest_log=str(output),
                pytest_sha256=file_hash(output),gpu_verified=False,jobs_submitted=False)
    atomic_json(root/'checks/cpu.json',report)
    if result.returncode:raise RuntimeError('CPU tests failed: '+str(output))
    print(json.dumps(report))


REFERENCE_ONLY_CHANGES = frozenset({
    'src/vpp2/flexible_mask/gradient_probe.py',
    'src/vpp2/flexible_mask/preflight.py',
    'tests/flexible_mask/test_flexible_contracts.py',
    'scripts/flexible_mask/check_zero_gradient.py',
})


def validate_reference_sources(previous,current):
    changed={name for name in previous.keys()|current.keys() if previous.get(name)!=current.get(name)}
    if not changed or not changed <= REFERENCE_ONLY_CHANGES:
        raise ValueError('Prior GPU evidence does not cover changed training code: '+str(sorted(changed)))
    return sorted(changed)


def reference_probe_identity(config):
    root=Path(config['root']);p=root/'checks/reference_evidence.json'
    if not p.exists():return identity(config)
    evidence=json.loads(p.read_text())
    if evidence['identity']!=identity(config):raise ValueError('Stale reference evidence')
    previous_path=Path(evidence['source_manifest'])
    if file_hash(previous_path)!=evidence['source_manifest_sha256']:raise ValueError('Reference manifest changed')
    previous=json.loads(previous_path.read_text());current=json.loads((root/'source_manifest.json').read_text())
    validate_reference_sources(previous['files'],current['files'])
    old_inputs=json.loads(Path(evidence['inputs']).read_text());new_inputs=json.loads((root/'inputs.json').read_text())
    if old_inputs.pop('identity')!=evidence['executed_identity'] or new_inputs.pop('identity')!=identity(config) or old_inputs!=new_inputs:
        raise ValueError('Prior verification used different model/data inputs')
    return evidence['executed_identity']


def finish_gpu(config):
    root=Path(config['root']);ident=identity(config);records={}
    require_receipt(config,'structure')
    reference_identity=reference_probe_identity(config)
    for name in ('uninterrupted','resumed','gradient_probe'):
        executed_identity=ident if name=='gradient_probe' else reference_identity
        folder=root/('gpu_probe_'+name)
        c=json.loads((folder/'contracts.json').read_text())
        if c['identity']!=executed_identity or not c.get('passed'):raise ValueError('Missing model contracts')
        values=[]
        for rank in range(8):
            r=json.loads((folder/f'fingerprint_rank{rank}.json').read_text())
            if r['identity']!=executed_identity or r['update']!=3:raise ValueError('Incomplete eight-GPU probe')
            initial=json.loads((folder/f'initial_fingerprint_rank{rank}.json').read_text())
            if initial['identity']!=executed_identity:raise ValueError('Initial probe identity changed')
            if any(initial['fingerprint'][key]==r['fingerprint'][key] for key in ('video','masks')):
                raise RuntimeError('Video weights or masks did not train')
            values.append(r['fingerprint'])
        records[name]=values
        metrics=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        for row in metrics:
            if row['video_gradient_norm']<=1e-12 or any(row['gradient/'+f]<=1e-12 for f in ('hidden','ffn','cross')):
                raise RuntimeError('Inactive video/mask gradient in GPU preflight')
            if name=='gradient_probe' and any(row['calibration/'+f+'/gt_norm']<=1e-12 for f in ('hidden','ffn','cross')):
                raise RuntimeError('Ground-truth action gradient does not reach every mask family')
    if records['uninterrupted']!=records['resumed']:
        raise RuntimeError('Interrupted/resumed state is not exact')
    if records['uninterrupted']!=records['gradient_probe']:
        raise RuntimeError('Gradient measurement changed the actual update')
    # Record each executed check, never infer numerical surgery from these probes.
    atomic_json(root/'checks/gpu.json',dict(identity=ident,passed=True,world_size=8,micro_batch=config['train']['micro_batch'],
        global_batch=256,exact_restart=True,diagnostic_gradient_noninterference=True,
        frozen_reference_unchanged=True,contracts=True,export_equivalence_still_required=True,
        reference_executed_identity=reference_identity,gradient_probe_executed_identity=ident))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--mode',choices=['cpu','probe','finish-gpu'],required=True)
    p.add_argument('--probe-name',default='uninterrupted');p.add_argument('--updates',type=int,default=3)
    a=p.parse_args();config=load(a.config)
    if a.mode=='cpu':cpu(config)
    elif a.mode=='finish-gpu':finish_gpu(config)
    else:
        from .train import run
        run(config,probe=True,smoke_updates=a.updates,probe_name=a.probe_name)

if __name__=='__main__':main()
