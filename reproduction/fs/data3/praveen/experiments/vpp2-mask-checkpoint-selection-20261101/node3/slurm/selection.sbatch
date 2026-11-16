#!/usr/bin/env bash
#SBATCH --job-name=mask-selection-fast8
#SBATCH --partition=eval
#SBATCH --nodelist=bagel-gpu-03
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --gres=none
#SBATCH --tres-per-task=gres/shard:1
#SBATCH --gres-flags=one-task-per-sharing
#SBATCH --time=03:00:00
#SBATCH --output=/data3/praveen/experiments/vpp2-mask-checkpoint-selection-20261101/node3/logs/%x-%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 VPP2_LOAD_META=1
export PYTHONPATH=/data3/praveen/experiments/vpp2-mask-checkpoint-selection-20261101/node3/source/src:/data3/praveen/experiments/vpp2-mask-checkpoint-selection-20261101/node3/code:/data3/praveen/experiments/vpp2-adaptive-latekv-mask-search-20261030/python-deps
cd /data3/praveen/experiments/vpp2-mask-checkpoint-selection-20261101/node3
printf 'host=%s job=%s visible=%s\n' "$(hostname)" "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES"
/data1/praveen/experiments/internw0-vpp2-distill-20261020/env/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 code/evaluate_remaining.py --root /data3/praveen/experiments/vpp2-mask-checkpoint-selection-20261101/node3
/data1/praveen/experiments/internw0-vpp2-distill-20261020/env/bin/python code/summarize.py --root /data3/praveen/experiments/vpp2-mask-checkpoint-selection-20261101/node3
rsync -a reports/ /data3/praveen/experiments/vpp2-mask-checkpoint-selection-20261101/reports/
