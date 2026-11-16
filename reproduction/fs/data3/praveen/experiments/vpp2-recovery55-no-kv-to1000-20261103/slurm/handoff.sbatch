#!/usr/bin/env bash
#SBATCH --job-name=no-kv-1000-handoff
#SBATCH --partition=train
#SBATCH --nodelist=bagel-gpu-03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --time=00:10:00
#SBATCH --output=/data3/praveen/experiments/vpp2-recovery55-no-kv-to1000-20261103/logs/%x-%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/data3/praveen/experiments/vpp2-recovery55-no-kv-to1000-20261103/source/src:/data3/praveen/experiments/vpp2-adaptive-latekv-mask-search-20261030/python-deps
/data1/praveen/experiments/internw0-vpp2-distill-20261020/env/bin/python -m vpp2.flexible_mask.recovery_extension --config /data3/praveen/experiments/vpp2-recovery55-no-kv-to1000-20261103/config.json --mode bind
