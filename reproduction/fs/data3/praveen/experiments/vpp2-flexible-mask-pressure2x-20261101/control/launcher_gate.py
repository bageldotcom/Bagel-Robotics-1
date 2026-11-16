"""CPU-only wait before an approved continuation can load any GPU model."""
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

ROOT = Path('/data3/praveen/experiments/vpp2-flexible-mask-pressure2x-20261101')
REVIEW = Path('/data3/praveen/experiments/vpp2-flexible-mask-execution-review-20261101')


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def is_waiting_job(job_id, root=ROOT, review=REVIEW):
    """Verify a specific live launcher is blocked before model initialization."""
    try:
        gate = read(root / 'control/launch_gate.json')
        if job_id not in gate['managed_jobs'] or job_id in gate['releases']:
            return False
        marker = read(root / 'control/launcher_wait' / (job_id + '.json'))
        if marker['job_id'] != job_id or marker['phase'] != 'waiting_for_pressure_handoff':
            return False
        if abs(time.time() - marker['heartbeat']) > 30:
            return False
        command = Path('/proc') / str(marker['pid']) / 'cmdline'
        if str(review / 'control/prepare_resume.py').encode() not in command.read_bytes().split(b'\0'):
            return False
        # All managed launchers run as the owning user on node 3.
        return command.stat().st_uid == os.getuid()
    except (OSError, ValueError, KeyError):
        return False


def main():
    job_id = os.environ.get('SLURM_JOB_ID', '')
    gate_path = ROOT / 'control/launch_gate.json'
    if not gate_path.exists():
        return
    gate = read(gate_path)
    if job_id not in gate['managed_jobs']:
        return
    marker_path = ROOT / 'control/launcher_wait' / (job_id + '.json')
    started = time.monotonic()
    print('Waiting for verified pressure handoff before model loading: ' + job_id, flush=True)
    while True:
        gate = read(gate_path)
        release = gate['releases'].get(job_id)
        if release is not None:
            selection = read(REVIEW / 'control/production_selection.json')
            if selection['root'] != release['root']:
                raise ValueError('Continuation release and selected root disagree')
            atomic(marker_path, dict(job_id=job_id, pid=os.getpid(), phase='released',
                heartbeat=time.time(), root=release['root'], utc=datetime.now(timezone.utc).isoformat()))
            print('Verified continuation route released: ' + release['root'], flush=True)
            return
        atomic(marker_path, dict(job_id=job_id, pid=os.getpid(),
            phase='waiting_for_pressure_handoff', heartbeat=time.time()))
        if time.monotonic() - started > 3600:
            raise TimeoutError('Pressure handoff did not release this launcher within one hour')
        time.sleep(5)


if __name__ == '__main__':
    main()
