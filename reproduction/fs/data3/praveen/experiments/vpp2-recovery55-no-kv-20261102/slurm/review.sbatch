#!/usr/bin/env bash
#SBATCH --job-name=no-kv-review250
#SBATCH --partition=train
#SBATCH --nodelist=bagel-gpu-03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --gres=none
#SBATCH --time=00:10:00
#SBATCH --output=/data3/praveen/experiments/vpp2-recovery55-no-kv-20261102/logs/%x-%j.out
set -euo pipefail
/data1/praveen/experiments/internw0-vpp2-distill-20261020/env/bin/python /data3/praveen/experiments/vpp2-recovery55-no-kv-20261102/control/review_250.py
