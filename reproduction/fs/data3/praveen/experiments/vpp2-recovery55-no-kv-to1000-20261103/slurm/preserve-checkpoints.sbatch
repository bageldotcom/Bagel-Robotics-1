#!/usr/bin/env bash
#SBATCH --job-name=preserve-recovery-checkpoints
#SBATCH --partition=train
#SBATCH --nodelist=bagel-gpu-03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=shard:0
#SBATCH --time=24:00:00
#SBATCH --output=/data3/praveen/experiments/vpp2-recovery55-no-kv-to1000-20261103/logs/%x-%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
echo "host=$(hostname) job=$SLURM_JOB_ID CPU-only checkpoint preservation"
exec nice -n 19 ionice -c 2 -n 7 /data1/praveen/experiments/internw0-vpp2-distill-20261020/env/bin/python /home/praveen/vpp2-save-all-20261103/preserve_checkpoints.py --root /data3/praveen/experiments/vpp2-recovery55-no-kv-to1000-20261103 --policy-archive /data2/praveen/experiments/vpp2-recovery55-no-kv-to1000-20261103-policy-archive --training-job 52231
