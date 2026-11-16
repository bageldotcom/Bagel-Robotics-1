"""Pin source and write Slurm artifacts. No sbatch/scontrol/scancel calls."""
import argparse,hashlib,json,shutil,subprocess,shlex
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
c=json.loads(Path(a.config).read_text());root=Path(c['root']);root.mkdir(parents=True,exist_ok=True)
repo=Path(__file__).resolve().parents[2]
files={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ('src','scripts','tests','configs')
       for p in sorted((repo/directory).rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
signature=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest();source=root/('source-'+signature[:12])
if not source.exists():
    for directory in ('src','scripts','tests','configs'):
        shutil.copytree(repo/directory,source/directory,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
(root/'source_manifest.json').write_text(json.dumps(dict(source=str(source),sha256=signature,files=files),indent=2)+'\n')
config_path=root/'config.json'
if config_path.exists() and json.loads(config_path.read_text())!=c:raise ValueError('Refusing to change published recipe')
config_path.write_text(json.dumps(c,indent=2)+'\n')
python='/data1/praveen/experiments/internw0-vpp2-distill-20261020/env/bin/python'
for name in ('logs','slurm','control','checks'):(root/name).mkdir(exist_ok=True)
for mode,n,partition,limit,mem in [('structure',1,'smoke','03:00:00','192G'),('baseline',8,'smoke','03:00:00','512G'),('distributed',8,'smoke','03:00:00','512G'),
    ('train',8,'train','2-00:00:00','512G'),('select',8,'eval','12:00:00','512G'),
    ('export',8,'smoke','03:00:00','512G'),('verify-export',1,'smoke','03:00:00','192G'),
    ('recover',8,'train','2-00:00:00','512G')]:
    script=root/'slurm'/f'{mode}.sbatch'
    body=f'''#!/usr/bin/env bash
#SBATCH --job-name=flexible-mask-{mode}
#SBATCH --partition={partition}
#SBATCH --nodelist=bagel-gpu-03
#SBATCH --nodes=1
#SBATCH --ntasks={n}
#SBATCH --ntasks-per-node={n}
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem}
#SBATCH --gres=none
#SBATCH --tres-per-task=gres/shard:1
#SBATCH --gres-flags=one-task-per-sharing
#SBATCH --time={limit}
#SBATCH --output={root}/logs/%x-%j.out
#SBATCH --signal=B:USR1@1800
set -euo pipefail
export PYTHONPATH={shlex.quote(str(source/'src')+':'+c['python_deps'])}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 VPP2_LOAD_META=1
cd {shlex.quote(str(source))}
echo "host=$(hostname) job=${{SLURM_JOB_ID:-none}} GPUs=${{CUDA_VISIBLE_DEVICES:-none}} source={source}"
{python} {source}/scripts/flexible_mask/verify_launch.py --config {config_path} --gpus {n}
'''
    launch=f'{python} -m torch.distributed.run --standalone --nproc_per_node={n}' if n>1 else python
    cfg=f'--config {config_path}'
    if mode=='baseline':
        body+=f'exec {launch} -m vpp2.flexible_mask.preflight {cfg} --mode probe --probe-name uninterrupted --updates 3\n'
    elif mode=='distributed':
        comparison=(f'{python} -m vpp2.flexible_mask.performance {cfg}\n' if c.get('execution',{}).get('teacher_storage')=='replicated' else '')
        body+=f'''{launch} -m vpp2.flexible_mask.preflight {cfg} --mode probe --probe-name uninterrupted --updates 3
{comparison}{launch} -m vpp2.flexible_mask.preflight {cfg} --mode probe --probe-name resumed --updates 1
{launch} -m vpp2.flexible_mask.preflight {cfg} --mode probe --probe-name resumed --updates 3
{launch} -m vpp2.flexible_mask.preflight {cfg} --mode probe --probe-name gradient_probe --updates 3
{python} -m vpp2.flexible_mask.preflight {cfg} --mode finish-gpu
'''
    else:
        entry={'structure':'verify_export --initial','train':'train','select':'select','export':'export',
               'verify-export':'verify_export','recover':'recover'}[mode]
        command=f'{launch} -m vpp2.flexible_mask.{entry} {cfg}'
        if mode in ('train','recover'):
            body+=f'''request_checkpoint() {{
  printf '{{"job_id":"%s","reason":"Slurm wall-time warning"}}\\n' "$SLURM_JOB_ID" > {root}/control/stop_request.partial
  mv {root}/control/stop_request.partial {root}/control/stop_request.json
}}
trap request_checkpoint USR1
{command} &
FLEXIBLE_CHILD=$!
while true; do
  set +e
  wait "$FLEXIBLE_CHILD"
  FLEXIBLE_EXIT=$?
  set -e
  if ! kill -0 "$FLEXIBLE_CHILD" 2>/dev/null; then exit "$FLEXIBLE_EXIT"; fi
done
'''
        else:body+='exec '+command+'\n'
    script.write_text(body);subprocess.run(['bash','-n',str(script)],check=True)
print(json.dumps(dict(source=str(source),config=str(config_path),scripts_syntax_passed=True,jobs_submitted=False)))
