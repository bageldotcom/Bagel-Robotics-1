#!/usr/bin/env bash
#SBATCH --job-name=bagel-reference
#SBATCH --partition=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=2-00:00:00
#SBATCH --output=local/train-%j.log
set -euo pipefail
: "${BAGEL_CONFIG:?Set BAGEL_CONFIG to a research configuration}"
: "${BAGEL_STAGE:?Set BAGEL_STAGE to learn-masks, prune or recover}"
BAGEL_ACTION="${BAGEL_ACTION:-verify}"
python -c 'import os,socket; print("host=",socket.gethostname(),"job=",os.environ.get("SLURM_JOB_ID"),"GPUs=",os.environ.get("CUDA_VISIBLE_DEVICES"))'
python -m bagel_robotics "$BAGEL_STAGE" --config "$BAGEL_CONFIG" --action "$BAGEL_ACTION"
