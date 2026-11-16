"""One authorized, checkpoint-preserving pressure continuation; CPU-only Slurm job.

No training source is edited, no running job is killed, and no checkpoint is rewound.
A failed preparation resumes the original recipe from the completed checkpoint.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path('/data3/praveen/experiments/vpp2-flexible-mask-pressure2x-20261101')
PARENT = Path('/data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-speed4-micro4')
REVIEW = Path('/data3/praveen/experiments/vpp2-flexible-mask-execution-review-20261101')
PYTHON = '/data1/praveen/experiments/internw0-vpp2-distill-20261020/env/bin/python'
DEPS = '/data3/praveen/experiments/vpp2-adaptive-latekv-mask-search-20261030/python-deps'
PARENT_JOB, NEXT_JOB, BACKUP_JOB = '50254', '50255', '50256'
TARGET = 1250
SELECTION = REVIEW / 'control/production_selection.json'
ACTIVE = {'RUNNING', 'COMPLETING', 'CONFIGURING', 'SUSPENDED', 'RESIZING'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write('\n'); output.flush(); os.fsync(output.fileno())
    os.replace(tmp, path)


def latest_metric(root):
    path = Path(root) / 'mask/metrics.jsonl'
    if not path.exists():
        return None
    with path.open('rb') as stream:
        stream.seek(0, 2); end = stream.tell()
        stream.seek(max(0, end - 131072))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            item = json.loads(line)
            if type(item.get('update')) is int:
                return item
        except (ValueError, UnicodeDecodeError):
            pass
    return None


def completed_parent(root, minimum, parent_identity):
    root = Path(root)
    tag = (root / 'mask/checkpoints/latest').read_text().strip()
    if not re.fullmatch(r'update_\d{6}', tag):
        raise ValueError('Invalid latest checkpoint tag')
    receipt = read(root / 'mask/checkpoints' / tag / 'COMPLETE.json')
    status = read(root / 'mask/status.json')
    metric = latest_metric(root)
    update = receipt['update']
    if not minimum <= update <= 1750:
        raise ValueError('Completed checkpoint is outside the approved continuation range')
    if tag != f'update_{update:06d}' or receipt['stage'] != 'mask':
        raise ValueError('Unexpected checkpoint tag or stage')
    if receipt['identity'] != parent_identity or status['identity'] != parent_identity:
        raise ValueError('Parent checkpoint identity mismatch')
    if status['status'] != 'checkpointed_interruption' or status['update'] != update:
        raise ValueError('No matching graceful-stop status')
    if not metric or metric['update'] != update:
        raise ValueError('Refusing to rewind reported completed updates')
    return update, receipt


def check_first_updates(root, start, base):
    rows = []
    for line in (Path(root) / 'mask/metrics.jsonl').read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue  # The writer may have an incomplete final line.
        if row['update'] > start:
            rows.append(row)
    if len(rows) < 3:
        return None
    first = rows[:3]
    if [r['update'] for r in first] != list(range(start + 1, start + 4)):
        raise ValueError('Continuation skipped or repeated updates')
    for row in first:
        expected = base * (1 + min((row['update'] - start) / 250, 1))
        if not math.isclose(row['size_weight'], expected, rel_tol=2e-6, abs_tol=1e-10):
            raise ValueError('Unexpected applied size coefficient')
        if not math.isclose(row['teacher_video_weight'], .05, rel_tol=2e-6):
            raise ValueError('Teacher-video coefficient changed')
        if not math.isclose(row['late_kv_weight'], .05, rel_tol=2e-6):
            raise ValueError('K/V coefficient changed')
        if not all(math.isfinite(row[k]) for k in ('total', 'gt_action', 'size_weight')):
            raise ValueError('Nonfinite resumed objective')
    return [dict(update=r['update'], size_weight=r['size_weight'],
                 seconds_per_update=r['seconds_per_update']) for r in first]


def run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=kwargs.pop('timeout', 30), **kwargs)
    if result.returncode:
        raise RuntimeError(f'Command failed {command!r}: {result.stderr.strip()}')
    return result.stdout.strip()


def job(job_id):
    result = subprocess.run(['scontrol', 'show', 'job', '-o', job_id],
                            capture_output=True, text=True, timeout=30)
    if result.returncode == 0 and result.stdout.strip():
        return dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9]*)=(\S+)', result.stdout))
    # A completed job may have aged out of the controller; never treat an
    # unavailable scheduler as proof that it is safe to run a new trainer.
    rows = run(['sacct', '-X', '-n', '-P', '-j', job_id, '-o', 'JobIDRaw,State,ExitCode']).splitlines()
    for row in rows:
        fields = row.split('|')
        if fields[0] == job_id:
            return dict(JobId=job_id, JobState=fields[1].split()[0], ExitCode=fields[2])
    raise RuntimeError('Cannot establish Slurm state for ' + job_id)


def waiting_for_gate(job_id):
    from launcher_gate import is_waiting_job
    return is_waiting_job(job_id, ROOT, REVIEW)



def env():
    values = os.environ.copy()
    values.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  PYTHONPATH=str(ROOT / 'source/src') + ':' + DEPS)
    return values


class Handoff:
    def __init__(self, max_seconds):
        self.deadline = time.monotonic() + max_seconds
        self.state_path = ROOT / 'control/handoff_status.json'
        self.held = set()
        self.switched = False
        self.stopped_update = None
        self.original = None
        self.parent_identity = read(PARENT / 'inputs.json')['identity']

    def event(self, phase, **details):
        value = dict(utc=utc(), phase=phase, watcher_job=os.environ.get('SLURM_JOB_ID'),
                     parent_job=PARENT_JOB, next_job=NEXT_JOB, backup_job=BACKUP_JOB,
                     target_update=TARGET, gated_launchers=sorted(self.held), **details)
        atomic(self.state_path, value)
        with (ROOT / 'control/handoff_events.jsonl').open('a') as output:
            output.write(json.dumps(value, sort_keys=True) + '\n')
        print(json.dumps(value, sort_keys=True), flush=True)

    def pause(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Handoff monitoring time limit reached')
        time.sleep(5)

    def hold(self, job_id):
        state = job(job_id)
        if state['JobState'] != 'PENDING':
            raise RuntimeError('Expected pending continuation: ' + job_id)
        gate_path = ROOT / 'control/launch_gate.json'
        if gate_path.exists():
            gate = read(gate_path)
        else:
            gate = dict(schema='pressure-handoff-launcher-v1', managed_jobs=[NEXT_JOB, BACKUP_JOB],
                        releases={}, parent_job=PARENT_JOB, target_root=str(ROOT))
        if job_id in gate['releases']:
            raise RuntimeError('A launcher was already released: ' + job_id)
        atomic(gate_path, gate)
        self.held.add(job_id)

    def release(self, job_id):
        if job_id not in self.held:
            return
        gate_path = ROOT / 'control/launch_gate.json'
        gate = read(gate_path)
        gate['releases'][job_id] = dict(root=read(SELECTION)['root'], utc=utc())
        atomic(gate_path, gate)
        self.held.remove(job_id)

    def prepare(self):
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('The watcher must run as a CPU-only Slurm job')
        p = job(PARENT_JOB)
        if p['JobState'] != 'RUNNING' or p.get('JobName') != 'flexible-reviewed-train':
            raise RuntimeError('Expected parent trainer is not running')
        if p.get('NodeList') != 'bagel-gpu-03' or not p.get('UserId', '').startswith('praveen('):
            raise RuntimeError('Unexpected parent ownership or node')
        self.original = read(SELECTION)
        if self.original['root'] != str(PARENT):
            raise ValueError('Production routing has already changed')
        backup = ROOT / ('control/handoff_backup_' + os.environ['SLURM_JOB_ID'])
        backup.mkdir(exist_ok=True)
        if (backup / 'production_selection.json').exists():
            raise ValueError('A previous activation backup exists; review before restarting watcher')
        atomic(backup / 'production_selection.json', self.original)
        for name in ('launch_selection.json', 'submitted_jobs.json'):
            atomic(backup / name, read(REVIEW / 'control' / name))
        atomic(backup / 'stop_request.json', read(PARENT / 'control/stop_request.json'))
        for jid, dependency in [(NEXT_JOB, PARENT_JOB), (BACKUP_JOB, NEXT_JOB)]:
            j = job(jid)
            if j['JobState'] != 'PENDING' or 'afterok:' + dependency not in j.get('Dependency', ''):
                raise ValueError('Unexpected continuation chain: ' + jid)
            if j.get('Priority') == '0':
                raise ValueError('Continuation was already held by another controller')
            (backup / f'job_{jid}.txt').write_text(run(['scontrol', 'show', 'job', jid]) + '\n')
        (backup / f'job_{PARENT_JOB}.txt').write_text(run(['scontrol', 'show', 'job', PARENT_JOB]) + '\n')
        self.hold(NEXT_JOB); self.hold(BACKUP_JOB)
        self.event('waiting_for_checkpoint_boundary', current_update=latest_metric(PARENT)['update'],
                   objective_unchanged_until_resume=True)

    def stop_and_bind(self):
        last_report = -1
        while True:
            state = job(PARENT_JOB)['JobState']
            metric = latest_metric(PARENT)
            if state not in ACTIVE:
                raise RuntimeError('Parent ended before its requested graceful checkpoint')
            if metric and metric['update'] >= TARGET:
                if metric['update'] > 1749:
                    raise ValueError('Approval window exceeded; refusing a late ramp')
                break
            if metric and metric['update'] // 10 != last_report:
                last_report = metric['update'] // 10
                self.event('waiting_for_checkpoint_boundary', current_update=metric['update'])
            self.pause()
        # Wait until update 1250 has actually been reported; writing after 1249
        # could race with that update's stop check and save an early checkpoint.
        atomic(PARENT / 'control/stop_request.json', dict(job_id=PARENT_JOB,
            reason='User-approved checkpoint and gradual 2x size-pressure continuation',
            requested_after_update=metric['update'], continuation_root=str(ROOT), utc=utc()))
        self.event('graceful_checkpoint_requested', current_update=metric['update'])
        while job(PARENT_JOB)['JobState'] in ACTIVE:
            self.pause()
        final_job = job(PARENT_JOB)
        if final_job['JobState'] != 'COMPLETED' or final_job.get('ExitCode') != '0:0':
            raise RuntimeError('Parent did not exit cleanly: ' + json.dumps(final_job))
        self.stopped_update, receipt = completed_parent(PARENT, TARGET, self.parent_identity)
        self.event('binding_completed_checkpoint', parent_update=self.stopped_update,
                   checkpoint_identity=receipt['checkpoint_identity'])
        command = [PYTHON, str(ROOT / 'control/bind_checkpoint.py'), '--parent-update', str(self.stopped_update)]
        with (ROOT / 'logs/bind_checkpoint.log').open('w') as output:
            subprocess.run(command, env=env(), stdout=output, stderr=subprocess.STDOUT,
                           check=True, timeout=1200)
        bound = read(ROOT / 'control/status.json')
        if bound['status'] != 'bound_awaiting_gpu_startup_check' or bound['parent_update'] != self.stopped_update:
            raise ValueError('Checkpoint binder did not produce a valid continuation')
        if read(SELECTION) != self.original:
            raise ValueError('Another controller changed production routing during handoff')
        chosen = dict(root=str(ROOT), selected='pressure2x', identity=bound['identity'],
                      parent_root=str(PARENT), parent_update=self.stopped_update,
                      checkpoint_frequency=250, global_batch=256, stopping_criteria_unchanged=True,
                      reason='User-approved 250-update ramp of size coefficient to 2x; other settings preserved',
                      authorization='User yes to checkpoint-and-resume near update 1250', utc=utc())
        atomic(SELECTION, chosen); self.switched = True
        # Exercise the exact routing code while both continuations are still held.
        output = run([PYTHON, str(REVIEW / 'control/prepare_resume.py')], env=env(), timeout=180)
        (ROOT / 'logs/prepare_resume.log').write_text(output + '\n')
        selected = read(REVIEW / 'control/launch_selection.json')
        if selected['root'] != str(ROOT) or selected['source'] != str(ROOT / 'source') or selected['finished']:
            raise ValueError('Existing launcher did not select the bound continuation')
        self.release(NEXT_JOB)
        self.event('continuation_released', parent_update=self.stopped_update,
                   ramp_end_update=self.stopped_update + 250, backup_held_until_gpu_check=True)

    def verify_resume(self):
        config = read(ROOT / 'config.json')
        base = config['pressure_continuation']['base_coefficient']
        last_phase = None
        while True:
            j = job(NEXT_JOB)
            if j['JobState'] not in ACTIVE | {'PENDING'}:
                raise RuntimeError('Continuation exited before verification: ' + json.dumps(j))
            receipt_path = ROOT / 'checks/pressure_gpu.json'
            if receipt_path.exists():
                receipt = read(receipt_path)
                if receipt.get('passed') is not True or receipt['update'] != self.stopped_update:
                    raise ValueError('Startup GPU gradient check failed or has wrong cursor')
                if receipt['identity'] != read(ROOT / 'inputs.json')['identity'] or len(receipt['reports']) != 8:
                    raise ValueError('Incomplete all-rank GPU verification')
                first = check_first_updates(ROOT, self.stopped_update, base)
                if first is not None:
                    self.release(BACKUP_JOB)
                    self.event('resumed_verified', parent_update=self.stopped_update,
                        first_updates=first, resumed_job=NEXT_JOB,
                        gpu_receipt=str(receipt_path), ramp_end_update=self.stopped_update + 250,
                        base_coefficient=base, final_coefficient=2 * base,
                        automatic_second_increase=False, stopping_criteria_unchanged=True)
                    atomic(ROOT / 'control/activation_result.json', read(self.state_path))
                    return
            phase = 'awaiting_gpu_startup_and_updates' if j['JobState'] in ACTIVE else 'waiting_for_allocation'
            if phase != last_phase:
                self.event(phase, parent_update=self.stopped_update, slurm_state=j['JobState'])
                last_phase = phase
            self.pause()

    def recover(self, error):
        details = dict(error=repr(error), recovery_errors=[])
        try:
            if self.original is None:
                self.event('handoff_failed_before_changes', **details); return
            child = latest_metric(ROOT)
            progressed = self.stopped_update is not None and child and child['update'] > self.stopped_update
            child_state = job(NEXT_JOB)['JobState']
            if self.switched and (progressed or (child_state in ACTIVE and not waiting_for_gate(NEXT_JOB))):
                # Never rewind or change a running continuation on ambiguous evidence.
                self.event('needs_review_training_or_progress_preserved',
                           child_update=child['update'] if child else None, **details)
                return
            current = read(SELECTION)
            if current['root'] not in (str(PARENT), str(ROOT)):
                raise RuntimeError('Unknown routing owner; not overriding it')
            atomic(SELECTION, self.original)
            if child_state == 'PENDING' or waiting_for_gate(NEXT_JOB):
                self.release(NEXT_JOB)
            elif self.switched and self.stopped_update is not None:
                # The first continuation failed before completing a new update.
                # Reuse the preserved backup job to resume the unmodified parent.
                if job(BACKUP_JOB)['JobState'] != 'PENDING':
                    raise RuntimeError('No pending fallback job remains')
                fallback = run(['sbatch', '--parsable', str(REVIEW / 'slurm/reviewed_train.sbatch')])
                details['fallback_job'] = fallback.split(';')[0]
                details['old_backup_dependency'] = 'afterok dependency remains blocked by failed first continuation'
            self.release(BACKUP_JOB)
            self.event('original_route_restored', parent_update=self.stopped_update, **details)
        except BaseException as recovery_error:
            details['recovery_errors'].append(repr(recovery_error))
            self.event('handoff_failed_requires_review', parent_update=self.stopped_update, **details)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-seconds', type=int, default=16200)
    args = parser.parse_args()
    (ROOT / 'control').mkdir(exist_ok=True)
    lock = (ROOT / 'control/handoff.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def interrupted(signum, _frame):
        raise RuntimeError('Watcher interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    h = Handoff(args.max_seconds)
    try:
        h.prepare(); h.stop_and_bind(); h.verify_resume()
    except BaseException as error:
        h.recover(error)
        raise


if __name__ == '__main__':
    main()
