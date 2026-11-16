#!/usr/bin/env bash
#SBATCH --job-name=recovery-prepare
#SBATCH --partition=eval
#SBATCH --nodelist=bagel-gpu-03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=none
#SBATCH --time=01:00:00
#SBATCH --output=/data3/praveen/experiments/vpp2-selected-mask-recovery-20261101/logs/%x-%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/data3/praveen/experiments/vpp2-selected-mask-recovery-20261101/source/src:/data3/praveen/experiments/vpp2-adaptive-latekv-mask-search-20261030/python-deps
export CUBLAS_WORKSPACE_CONFIG=:4096:8 VPP2_LOAD_META=1
cd /data3/praveen/experiments/vpp2-selected-mask-recovery-20261101/source
printf 'host=%s job=%s visible=%s\n' "$(hostname)" "$SLURM_JOB_ID" "${CUDA_VISIBLE_DEVICES:-none}"
export CUDA_VISIBLE_DEVICES=""
/data1/praveen/experiments/internw0-vpp2-distill-20261020/env/bin/python /data3/praveen/experiments/vpp2-selected-mask-recovery-20261101/control/prepare_cpu.py
