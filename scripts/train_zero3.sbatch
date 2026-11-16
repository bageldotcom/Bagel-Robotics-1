#!/usr/bin/env bash
#SBATCH --job-name=bagel-no-kv
#SBATCH --partition=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1T
#SBATCH --gres=gpu:8
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=outputs/slurm-%x-%j.log
set -euo pipefail
: "${BAGEL_REPO:?Set BAGEL_REPO to this checkout}"
: "${BAGEL_WORKSPACE:?Set BAGEL_WORKSPACE to an initialized workspace}"
: "${BAGEL_TRAIN_IMAGE:?Set BAGEL_TRAIN_IMAGE to the built container}"
: "${BAGEL_STAGE:?Set BAGEL_STAGE to the desired released stage}"
BAGEL_ACTION="${BAGEL_ACTION:-verify}"
cd "$BAGEL_REPO"
python -c 'import os,socket; print("host=",socket.gethostname(),"job=",os.environ.get("SLURM_JOB_ID"),"GPUs=",os.environ.get("CUDA_VISIBLE_DEVICES"))'
git rev-parse HEAD
request_checkpoint() {
  python run.py request-stop --workspace "$BAGEL_WORKSPACE" --stage "$BAGEL_STAGE" --job-id "$SLURM_JOB_ID"
}
trap request_checkpoint USR1
python scripts/train.py --workspace "$BAGEL_WORKSPACE" --stage "$BAGEL_STAGE" \
  --image "$BAGEL_TRAIN_IMAGE" --action "$BAGEL_ACTION" &
BAGEL_CHILD=$!
while true; do
  set +e
  wait "$BAGEL_CHILD"
  BAGEL_EXIT=$?
  set -e
  if ! kill -0 "$BAGEL_CHILD" 2>/dev/null; then exit "$BAGEL_EXIT"; fi
done
