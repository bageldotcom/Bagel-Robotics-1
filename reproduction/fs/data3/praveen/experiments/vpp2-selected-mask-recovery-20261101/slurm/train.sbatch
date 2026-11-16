#!/usr/bin/env bash
#SBATCH --job-name=selected-mask-recovery
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
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=/data3/praveen/experiments/vpp2-selected-mask-recovery-20261101/logs/%x-%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/data3/praveen/experiments/vpp2-selected-mask-recovery-20261101/source/src:/data3/praveen/experiments/vpp2-adaptive-latekv-mask-search-20261030/python-deps
export CUBLAS_WORKSPACE_CONFIG=:4096:8 VPP2_LOAD_META=1
cd /data3/praveen/experiments/vpp2-selected-mask-recovery-20261101/source
printf 'host=%s job=%s visible=%s\n' "$(hostname)" "$SLURM_JOB_ID" "${CUDA_VISIBLE_DEVICES:-none}"

request_checkpoint() {
  printf '{"job_id":"%s","reason":"Slurm wall-time warning"}\n' "$SLURM_JOB_ID" > /data3/praveen/experiments/vpp2-selected-mask-recovery-20261101/control/stop_request.partial
  mv /data3/praveen/experiments/vpp2-selected-mask-recovery-20261101/control/stop_request.partial /data3/praveen/experiments/vpp2-selected-mask-recovery-20261101/control/stop_request.json
}
trap request_checkpoint USR1
/data1/praveen/experiments/internw0-vpp2-distill-20261020/env/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 -m vpp2.flexible_mask.recover --config /data3/praveen/experiments/vpp2-selected-mask-recovery-20261101/config.json &
RECOVERY_CHILD=$!
while true; do
  set +e
  wait "$RECOVERY_CHILD"
  RECOVERY_EXIT=$?
  set -e
  if ! kill -0 "$RECOVERY_CHILD" 2>/dev/null; then exit "$RECOVERY_EXIT"; fi
done
