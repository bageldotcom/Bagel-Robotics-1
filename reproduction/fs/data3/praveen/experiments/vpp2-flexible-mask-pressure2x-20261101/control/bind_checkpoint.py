"""Bind a complete checkpoint after an approved handoff. Never stops jobs."""
import argparse
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import torch
import zstandard
from vpp2.flexible_mask.config import load, identity, sources
from vpp2.flexible_mask.continuation import validate_parent
from vpp2.flexible_mask.preflight import cpu
from vpp2.flexible_mask.pressure import validate_parent_state, require_runtime_evidence, published_sources
from vpp2.full_mask.panel import file_hash
from vpp2.compact_kv.spec import atomic_json

PARENT=Path('/data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-speed4-micro4')
ROOT=Path('/data3/praveen/experiments/vpp2-flexible-mask-pressure2x-20261101')


def client_state(folder):
    paths=sorted(folder.glob('*model_states.pt'))
    if not paths:raise ValueError('No model/client-state shard')
    with paths[0].open('rb') as raw,zstandard.ZstdDecompressor().stream_reader(raw) as stream:
        data=stream.read(64*1024**2+1)
    if len(data)>64*1024**2:raise ValueError('Unexpectedly large client-state shard')
    state=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
    if not all(k in state for k in ('update','cursor','coefficient','masks','mask_optimizer','rank_rng')):
        raise ValueError('Checkpoint lacks complete continuation state')
    return state


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent-update',type=int,required=True)
    args=parser.parse_args()
    if not 1250<=args.parent_update<=1750:raise ValueError('Expected completed update between 1250 and 1750')
    running=subprocess.run(['squeue','-h','-u','praveen','-t','RUNNING','-o','%i|%j'],
                           check=True,capture_output=True,text=True).stdout
    from launcher_gate import is_waiting_job
    for row in running.splitlines():
        job_id, name = row.split('|', 1)
        if ('flexible-reviewed-train' in name or 'flexible-pressure2x' in name) and not is_waiting_job(job_id):
            raise RuntimeError('Training is active: only a verified pre-model launcher wait is allowed')
    old=load(PARENT/'config.json');tag=f'update_{args.parent_update:06d}'
    folder=PARENT/'mask/checkpoints'/tag
    if (PARENT/'mask/checkpoints/latest').read_text().strip()!=tag:
        raise ValueError('Not the latest checkpoint; refusing to discard progress')
    receipt=json.loads((folder/'COMPLETE.json').read_text())
    if receipt['update']!=args.parent_update or receipt['stage']!='mask':raise ValueError('Wrong checkpoint')
    parent_inputs=json.loads((PARENT/'inputs.json').read_text())
    if receipt['identity']!=parent_inputs['identity']:raise ValueError('Parent identities disagree')
    state=client_state(folder);base=state['coefficient']['applied']
    c=copy.deepcopy(old);c['root']=str(ROOT)
    c['wandb']['name']='GT1-video005-lateKV005-size-pressure2x-continuation'
    c['pressure_continuation']=dict(schema='vpp2-size-pressure-ramp-v1',start_update=args.parent_update,
        ramp_updates=250,target_multiplier=2.,base_coefficient=base,absolute_cap=2*base,
        target_video_parameters=10_000_000_000,automatic_second_increase=False)
    c['resume_from']=dict(root=str(PARENT),tag=tag,identity=receipt['identity'],
        config_sha256=file_hash(PARENT/'config.json'),receipt_sha256=file_hash(folder/'COMPLETE.json'))
    validate_parent_state(state,c)
    for name in ('control/evidence','checks','logs','mask/diagnostics','preserved_parent/checkpoints'):
        (ROOT/name).mkdir(parents=True,exist_ok=True)
    path=ROOT/'config.json'
    if path.exists() and json.loads(path.read_text())!=c:raise ValueError('Target already bound differently')
    atomic_json(path,c);load(path)
    parent_manifest=json.loads((PARENT/'source_manifest.json').read_text())
    for name,sha in parent_manifest['files'].items():
        if file_hash(Path(parent_manifest['source'])/name)!=sha:raise ValueError('Parent source changed: '+name)
    current_sources=published_sources(c)
    inputs=copy.deepcopy(parent_inputs);inputs['identity']=identity(c)
    atomic_json(ROOT/'inputs.json',inputs);validate_parent(c,'mask')
    pinned=ROOT/'preserved_parent/checkpoints'/tag;pinned.mkdir(exist_ok=True)
    for name,info in receipt['files'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe shard path')
        source=folder/relative;target=pinned/relative
        if source.stat().st_size!=info['size']:raise ValueError('Incomplete shard: '+name)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if not os.path.samefile(source,target):raise ValueError('Conflicting preserved checkpoint')
        else:os.link(source,target)
    shutil.copyfile(folder/'COMPLETE.json',pinned/'COMPLETE.json')
    atomic_json(ROOT/'control/parent_state.json',dict(update=state['update'],cursor=state['cursor'],
        coefficient=state['coefficient'],checkpoint_identity=receipt['checkpoint_identity'],
        preserved_checkpoint=str(pinned),parent_best=state.get('best',[]),
        optimizer_state_preserved=True,rng_state_preserved=True,
        runtime_will_verify_all_checkpoint_shard_hashes=True))
    # Preserve parent observations explicitly as inherited history.
    for p in sorted((PARENT/'mask/diagnostics').glob('*.json')):
        shutil.copyfile(p,ROOT/'mask/diagnostics'/p.name)
    with (ROOT/'mask/metrics.jsonl').open('w') as output:
        for line in (PARENT/'mask/metrics.jsonl').open():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if r.get('update',0)<=state['update']:output.write(json.dumps(r)+'\n')
    cpu(c)
    evidence=[];copied={}
    for name,source in [('parent_gpu',PARENT/'checks/gpu.json'),('parent_manifest',PARENT/'source_manifest.json'),
                        ('parent_config',PARENT/'config.json'),('parent_receipt',folder/'COMPLETE.json')]:
        target=ROOT/'control/evidence'/f'{name}.json';shutil.copyfile(source,target);copied[name]=str(target)
        evidence.append(dict(path=str(target),sha256=file_hash(target)))
    changed=sorted(k for k in parent_manifest['files'].keys()|current_sources.keys()
                   if parent_manifest['files'].get(k)!=current_sources.get(k))
    atomic_json(ROOT/'checks/pressure_prerequisites.json',dict(identity=identity(c),passed=True,evidence=evidence,
        parent_manifest=copied['parent_manifest'],parent_gpu_receipt=copied['parent_gpu'],
        changed_source_files=changed,runtime_kernels_and_optimizer_unchanged=True,
        new_full_policy_gpu_trial_executed=False,gradient_check_required_before_first_update=True))
    require_runtime_evidence(c)
    atomic_json(ROOT/'control/status.json',dict(status='bound_awaiting_gpu_startup_check',identity=identity(c),
        parent_update=state['update'],ramp_end=state['update']+250,
        first_possible_stable_check=((state['update']+250+249)//250)*250+1000,
        jobs_submitted=False,current_jobs_modified=False))
    print(json.dumps(dict(config=str(path),identity=identity(c),parent_update=state['update'],
                         base_coefficient=base,final_coefficient=2*base)))

if __name__=='__main__':main()
