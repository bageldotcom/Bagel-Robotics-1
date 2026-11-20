"""Evidence-bound readiness. CPU checks never grant GPU training readiness."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch
from bagel_robotics.adaptive_pruning.config import load, identity
from bagel_robotics.data_selection.manifest import file_hash
from bagel_robotics._internal.compact_kv.spec import atomic_json


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
    from bagel_robotics._internal.training.rng import rng_state
    return {name:fingerprint(value) for name,value in dict(
        video={n:getattr(p,'ds_tensor',p) for n,p in runtime.engine.module.named_parameters()},
        optimizer=runtime.engine.optimizer.state_dict(),
        masks=None if runtime.masks is None else runtime.masks.state_dict(),
        mask_optimizer=None if runtime.mask_optimizer is None else runtime.mask_optimizer.state_dict(),
        rng=rng_state()).items()}


def gpu_contracts(runtime,panel):
    from bagel_robotics._internal.adaptive_mask.data import prepare
    from bagel_robotics._internal.training.rng import diagnostic_rng
    from bagel_robotics._internal.adaptive_mask.equivalence import compare
    from bagel_robotics.adaptive_pruning.runtime import independent_storage
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
    root=Path(config['root'])
    inputs=json.loads((root/'inputs.json').read_text())
    if inputs['identity']!=identity(config):raise ValueError('Prepared inputs/source identity changed')
    output=root/'checks/pytest.txt'
    with output.open('w') as f:
        result=subprocess.run([sys.executable,'-m','bagel_robotics.selftest'],stdout=f,stderr=subprocess.STDOUT)
    report=dict(identity=identity(config),passed=result.returncode==0,pytest_log=str(output),
                pytest_sha256=file_hash(output),gpu_verified=False,jobs_submitted=False)
    atomic_json(root/'checks/cpu.json',report)
    if result.returncode:raise RuntimeError('CPU tests failed: '+str(output))
    print(json.dumps(report))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--mode',choices=['cpu','probe'],required=True)
    parser.add_argument('--probe-name',default='smoke');parser.add_argument('--updates',type=int,default=1)
    args=parser.parse_args();config=load(args.config)
    if args.mode=='cpu':cpu(config)
    else:
        if args.updates<1:parser.error('--updates must be positive')
        from bagel_robotics.adaptive_pruning.train import run
        run(config,stage=config['stage'],probe=True,smoke_updates=args.updates,probe_name=args.probe_name)

if __name__=='__main__':main()
