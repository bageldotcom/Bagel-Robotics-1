#!/usr/bin/env bash
#SBATCH --job-name=flexible-pressure2x
#SBATCH --partition=train
#SBATCH --nodelist=bagel-gpu-03
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --gres=none
#SBATCH --tres-per-task=gres/shard:1
#SBATCH --gres-flags=one-task-per-sharing
#SBATCH --time=2-00:00:00
#SBATCH --output=/data3/praveen/experiments/vpp2-flexible-mask-pressure2x-20261101/logs/%x-%j.out
#SBATCH --signal=B:USR1@1800
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 VPP2_LOAD_META=1
PRESSURE_ROOT=/data3/praveen/experiments/vpp2-flexible-mask-pressure2x-20261101
PRESSURE_SOURCE="$PRESSURE_ROOT/source"
PRESSURE_CONFIG="$PRESSURE_ROOT/config.json"
PRESSURE_PYTHON=/data1/praveen/experiments/internw0-vpp2-distill-20261020/env/bin/python
export PYTHONPATH="$PRESSURE_SOURCE/src:/data3/praveen/experiments/vpp2-adaptive-latekv-mask-search-20261030/python-deps"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} GPUs=${CUDA_VISIBLE_DEVICES:-none}"
cd "$PRESSURE_SOURCE"
"$PRESSURE_PYTHON" "$PRESSURE_SOURCE/scripts/flexible_mask/verify_launch.py" --config "$PRESSURE_CONFIG" --gpus 8
request_checkpoint() {
  printf '{"job_id":"%s","reason":"Slurm wall-time warning"}\n' "$SLURM_JOB_ID" > "$PRESSURE_ROOT/control/stop_request.partial"
  mv "$PRESSURE_ROOT/control/stop_request.partial" "$PRESSURE_ROOT/control/stop_request.json"
}
trap request_checkpoint USR1
"$PRESSURE_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 -m vpp2.flexible_mask.train --config "$PRESSURE_CONFIG" &
PRESSURE_CHILD=$!
while true; do
  set +e
  wait "$PRESSURE_CHILD"
  PRESSURE_EXIT=$?
  set -e
  if ! kill -0 "$PRESSURE_CHILD" 2>/dev/null; then exit "$PRESSURE_EXIT"; fi
done
