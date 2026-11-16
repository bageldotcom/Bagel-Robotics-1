#!/usr/bin/env bash
#SBATCH --job-name=flexible-mask-select
#SBATCH --partition=eval
#SBATCH --nodelist=bagel-gpu-03
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --gres=none
#SBATCH --tres-per-task=gres/shard:1
#SBATCH --gres-flags=one-task-per-sharing
#SBATCH --time=12:00:00
#SBATCH --output=/data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-micro2/logs/%x-%j.out
#SBATCH --signal=B:USR1@1800
set -euo pipefail
export PYTHONPATH=/data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-micro2/source-aa7bf6c846b9/src:/data3/praveen/experiments/vpp2-adaptive-latekv-mask-search-20261030/python-deps
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 VPP2_LOAD_META=1
cd /data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-micro2/source-aa7bf6c846b9
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} GPUs=${CUDA_VISIBLE_DEVICES:-none} source=/data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-micro2/source-aa7bf6c846b9"
/data1/praveen/experiments/internw0-vpp2-distill-20261020/env/bin/python /data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-micro2/source-aa7bf6c846b9/scripts/flexible_mask/verify_launch.py --config /data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-micro2/config.json --gpus 8
exec /data1/praveen/experiments/internw0-vpp2-distill-20261020/env/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 -m vpp2.flexible_mask.select --config /data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-micro2/config.json
