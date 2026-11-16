#!/usr/bin/env bash
#SBATCH --job-name=pressure2x-handoff
#SBATCH --partition=smoke
#SBATCH --nodelist=bagel-gpu-03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=none
#SBATCH --time=03:00:00
#SBATCH --output=/data3/praveen/experiments/vpp2-flexible-mask-pressure2x-20261101/logs/%x-%j.out
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} CPU-only pressure handoff watcher"
exec /data1/praveen/experiments/internw0-vpp2-distill-20261020/env/bin/python /data3/praveen/experiments/vpp2-flexible-mask-pressure2x-20261101/control/watch_handoff.py --max-seconds 9000
