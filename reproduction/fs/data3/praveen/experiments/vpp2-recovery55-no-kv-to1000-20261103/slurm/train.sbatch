#!/usr/bin/env bash
#SBATCH --job-name=recovery-no-kv-to1000
#SBATCH --partition=train
#SBATCH --nodelist=bagel-gpu-03
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=1024G
#SBATCH --gres=none
#SBATCH --tres-per-task=gres/shard:1
#SBATCH --gres-flags=one-task-per-sharing
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=/data3/praveen/experiments/vpp2-recovery55-no-kv-to1000-20261103/logs/%x-%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/data3/praveen/experiments/vpp2-recovery55-no-kv-to1000-20261103/source/src:/data3/praveen/experiments/vpp2-adaptive-latekv-mask-search-20261030/python-deps
export CUBLAS_WORKSPACE_CONFIG=:4096:8 VPP2_LOAD_META=1
export VPP2_CHECKPOINT_LOAD_TMPDIR=/data3/praveen/experiments/vpp2-recovery55-no-kv-to1000-20261103/artifacts/load-scratch
cd /data3/praveen/experiments/vpp2-recovery55-no-kv-to1000-20261103/source
echo "host=$(hostname) job=$SLURM_JOB_ID visible=${CUDA_VISIBLE_DEVICES:-unset}"
NGPUS="$(/data1/praveen/experiments/internw0-vpp2-distill-20261020/env/bin/python -c 'import os; print(len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")))')"
test "$NGPUS" -eq 8
/data1/praveen/experiments/internw0-vpp2-distill-20261020/env/bin/python - <<'PYCHECK'
import importlib.metadata,torch
from vpp2.flexible_mask.config import load
from vpp2.flexible_mask.low_lr import verify_parent,check_storage
c=load('/data3/praveen/experiments/vpp2-recovery55-no-kv-to1000-20261103/config.json');verify_parent(c);check_storage(c)
from vpp2.flexible_mask.recovery_extension import bind_parent
bind_parent(c)
assert torch.__version__=='2.10.0+cu128' and torch.version.cuda=='12.8'
assert importlib.metadata.version('deepspeed')=='0.19.2'
PYCHECK
request_checkpoint() {
 /data1/praveen/experiments/internw0-vpp2-distill-20261020/env/bin/python - <<'PYSTOP'
import json,os
from pathlib import Path
p=Path('/data3/praveen/experiments/vpp2-recovery55-no-kv-to1000-20261103/control/stop_request.json');q=p.with_suffix('.partial')
q.write_text(json.dumps(dict(job_id=os.environ['SLURM_JOB_ID'],reason='No-K/V extension Slurm wall-time warning')));q.replace(p)
PYSTOP
}
trap request_checkpoint USR1
/data1/praveen/experiments/internw0-vpp2-distill-20261020/env/bin/python -m torch.distributed.run --standalone --nproc_per_node="$NGPUS" -m vpp2.flexible_mask.test_gradient_gpu --config /data3/praveen/experiments/vpp2-recovery55-no-kv-to1000-20261103/config.json
/data1/praveen/experiments/internw0-vpp2-distill-20261020/env/bin/python -m torch.distributed.run --standalone --nproc_per_node="$NGPUS" -m vpp2.flexible_mask.test_recovery_resume_gpu --config /data3/praveen/experiments/vpp2-recovery55-no-kv-to1000-20261103/config.json
/data1/praveen/experiments/internw0-vpp2-distill-20261020/env/bin/python -m torch.distributed.run --standalone --nproc_per_node="$NGPUS" -m vpp2.flexible_mask.recovery_extension_entry --config /data3/praveen/experiments/vpp2-recovery55-no-kv-to1000-20261103/config.json &
TRAIN_PID=$!
while true; do
 set +e
 wait "$TRAIN_PID"
 TRAIN_RC=$?
 set -e
 if ! kill -0 "$TRAIN_PID" 2>/dev/null; then exit "$TRAIN_RC"; fi
done
