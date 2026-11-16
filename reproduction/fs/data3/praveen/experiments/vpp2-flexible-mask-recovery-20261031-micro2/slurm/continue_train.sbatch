#!/usr/bin/env bash
#SBATCH --job-name=flexible-mask-continue
#SBATCH --partition=train
#SBATCH --nodelist=bagel-gpu-03
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --gres=none
#SBATCH --tres-per-task=gres/shard:1
#SBATCH --gres-flags=one-task-per-sharing
#SBATCH --time=2-00:00:00
#SBATCH --output=/data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-micro2/logs/%x-%j.out
#SBATCH --signal=B:USR1@1800
set -euo pipefail
CONTINUE_MODE=$(/data1/praveen/experiments/internw0-vpp2-distill-20261020/env/bin/python - <<'GUARD'
import json,re
from pathlib import Path
root=Path('/data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-micro2')
expected='72f7387ca10b6e2b2c08bcab5ad77ec8f9d7afea8ed69777a93a16a8f4ddd6da'
status=json.loads((root/'mask/status.json').read_text())
assert status['identity']==expected, 'Continuation identity mismatch'
if (status['status']=='ready_for_recovery' and status['update']>=2000) or (status['status']=='selection_not_converged' and status['update']==3000):
    print('finished')
else:
    assert status['status']=='checkpointed_interruption', 'Cannot continue failed or unresolved training'
    tag=(root/'mask/checkpoints/latest').read_text().strip()
    assert re.fullmatch(r'update_[0-9]{6}',tag)
    receipt=json.loads((root/'mask/checkpoints'/tag/'COMPLETE.json').read_text())
    assert receipt['identity']==expected and receipt['stage']=='mask' and receipt['update']==status['update']
    print('resume')
GUARD
)
if [[ "$CONTINUE_MODE" == finished ]]; then
    echo 'Mask learning already finished; no further training is needed.'
    exit 0
fi
if [[ "$CONTINUE_MODE" != resume ]]; then exit 2; fi
export PYTHONPATH=/data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-micro2/source-aa7bf6c846b9/src:/data3/praveen/experiments/vpp2-adaptive-latekv-mask-search-20261030/python-deps
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 VPP2_LOAD_META=1
cd /data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-micro2/source-aa7bf6c846b9
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} GPUs=${CUDA_VISIBLE_DEVICES:-none} source=/data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-micro2/source-aa7bf6c846b9"
/data1/praveen/experiments/internw0-vpp2-distill-20261020/env/bin/python /data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-micro2/source-aa7bf6c846b9/scripts/flexible_mask/verify_launch.py --config /data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-micro2/config.json --gpus 8
request_checkpoint() {
  printf '{"job_id":"%s","reason":"Slurm wall-time warning"}\n' "$SLURM_JOB_ID" > /data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-micro2/control/stop_request.partial
  mv /data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-micro2/control/stop_request.partial /data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-micro2/control/stop_request.json
}
trap request_checkpoint USR1
/data1/praveen/experiments/internw0-vpp2-distill-20261020/env/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 -m vpp2.flexible_mask.train --config /data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-micro2/config.json &
FLEXIBLE_CHILD=$!
while true; do
  set +e
  wait "$FLEXIBLE_CHILD"
  FLEXIBLE_EXIT=$?
  set -e
  if ! kill -0 "$FLEXIBLE_CHILD" 2>/dev/null; then exit "$FLEXIBLE_EXIT"; fi
done
