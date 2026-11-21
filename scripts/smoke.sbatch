#!/usr/bin/env bash
#SBATCH --job-name=bagel-inference-smoke
#SBATCH --partition=smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=local/smoke-%j.log
set -euo pipefail
: "${BAGEL_CONFIG:?Set BAGEL_CONFIG to an evaluation configuration}"
python -c 'import os,socket; print("host=",socket.gethostname(),"job=",os.environ.get("SLURM_JOB_ID"),"GPUs=",os.environ.get("CUDA_VISIBLE_DEVICES"))'
python -m bagel_robotics evaluate --config "$BAGEL_CONFIG" --operation smoke
