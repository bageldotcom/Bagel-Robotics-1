"""Read-only launch integrity/resource checks inside the allocated batch job."""
import argparse,json,os,shutil,hashlib
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--gpus',type=int,required=True);a=p.parse_args()
c=json.loads(Path(a.config).read_text());root=Path(c['root']);receipt=json.loads((root/'source_manifest.json').read_text())
import zstandard
if zstandard.__version__!='0.25.0':raise RuntimeError('Install pinned checkpoint dependency: zstandard==0.25.0')
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('No Slurm allocation')
if torch.cuda.device_count()!=a.gpus:raise ValueError('Allocated GPU count differs from launcher')
free_gib=[torch.cuda.mem_get_info(i)[0]/2**30 for i in range(a.gpus)]
minimum_free=76 if c.get('execution',{}).get('teacher_storage')=='replicated' else 68
if min(free_gib)<minimum_free:raise RuntimeError(f'Allocated GPUs need at least {minimum_free} GiB free before loading; existing workloads are left untouched')
if shutil.disk_usage(root).free<c['launch_policy']['minimum_disk_free_gib']*2**30:raise RuntimeError('Insufficient checkpoint storage')
source=Path(receipt['source'])
for name,expected in receipt['files'].items():
    if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:raise ValueError('Published source changed: '+name)
from vpp2.flexible_mask.preflight import require_receipt
require_receipt(c,'cpu')
from vpp2.full_mask.panel import file_hash
inputs=json.loads((root/'inputs.json').read_text())
for item in inputs['teacher'].values():
    if file_hash(item['path'])!=item['sha256']:raise ValueError('Frozen teacher checkpoint changed')
if file_hash(c['initial_masks'])!=c['initial_masks_sha256']:raise ValueError('Initial mask artifact changed')
print(json.dumps(dict(source_verified=True,gpus=a.gpus,free_disk_gib=shutil.disk_usage(root).free/2**30)))
