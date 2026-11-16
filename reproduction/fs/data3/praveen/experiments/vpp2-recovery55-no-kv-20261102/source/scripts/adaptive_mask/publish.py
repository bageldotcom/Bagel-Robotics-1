"""Freeze source/config and prepare validated Slurm scripts; never submit implicitly."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess
p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
c=json.loads(Path(a.config).read_text());root=Path(c['root']);home=Path(__file__).resolve().parents[2]
files={str(path.relative_to(home)):hashlib.sha256(path.read_bytes()).hexdigest() for base in ('src','configs','tests','scripts') for path in sorted((home/base).rglob('*')) if path.is_file() and '__pycache__' not in path.parts and '.pytest_cache' not in path.parts}
signature=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest();source=root/('source-'+signature[:12])
if not source.exists():
 for base in ('src','configs','tests','scripts'):shutil.copytree(home/base,source/base,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
manifest=dict(source=str(source),sha256=signature,files=files)
(root/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
python='/data1/praveen/experiments/internw0-vpp2-distill-20261020/env/bin/python'
for mode,n,partition,limit,memory in [('native',1,'smoke','03:00:00','160G'),('distributed',8,'smoke','03:00:00','1100G'),('train',8,'train','2-00:00:00','1100G'),('audit',8,'eval','12:00:00','1100G'),('final-export',1,'smoke','03:00:00','160G')]:
 script=root/'slurm'/f'{mode}.sbatch'
 content=f'''#!/usr/bin/env bash
#SBATCH --job-name=adaptive-hidden-{mode}
#SBATCH --partition={partition}
#SBATCH --nodelist=bagel-gpu-03
#SBATCH --nodes=1
#SBATCH --ntasks={n}
#SBATCH --ntasks-per-node={n}
#SBATCH --cpus-per-task=8
#SBATCH --mem={memory}
#SBATCH --gres=none
#SBATCH --tres-per-task=gres/shard:1
#SBATCH --gres-flags=one-task-per-sharing
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --time={limit}
#SBATCH --output={root}/logs/%x-%j.out
#SBATCH --signal=B:USR1@1800
set -euo pipefail
export PYTHONPATH={source}/src:{c['python_deps']}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
cd {source}
echo "host=$(hostname) job=${{SLURM_JOB_ID:-none}} GPUs=${{CUDA_VISIBLE_DEVICES:-none}} source={source}"
{python} {source}/scripts/adaptive_mask/verify_launch.py --config {root}/config.json --gpus {n} --mode {mode}
'''
 if mode=='train':
  content+=f'''request_checkpoint() {{
  printf '{{"job_id":"%s","reason":"Slurm wall-time warning"}}\\n' "$SLURM_JOB_ID" > {root}/control/stop_request.partial
  mv {root}/control/stop_request.partial {root}/control/stop_request.json
}}
trap request_checkpoint USR1
{python} -m torch.distributed.run --standalone --nproc_per_node={n} -m vpp2.adaptive_mask.train --config {root}/config.json &
ADAPTIVE_LAUNCH_PID=$!
while true; do
  set +e
  wait "$ADAPTIVE_LAUNCH_PID"
  ADAPTIVE_EXIT_CODE=$?
  set -e
  if ! kill -0 "$ADAPTIVE_LAUNCH_PID" 2>/dev/null; then exit "$ADAPTIVE_EXIT_CODE"; fi
done
'''
 else:
  launcher=f'{python} -m torch.distributed.run --standalone --nproc_per_node={n}' if n>1 else python
  entry=f'vpp2.adaptive_mask.audit --config {root}/config.json' if mode=='audit' else f'vpp2.adaptive_mask.preflight --config {root}/config.json --mode {mode}'
  content+=f'exec {launcher} -m {entry}\n'
 script.write_text(content);subprocess.run(['bash','-n',str(script)],check=True)
print(json.dumps(dict(source=str(source),manifest=str(root/'source_manifest.json'),scripts_syntax_passed=True,jobs_submitted=False)))
