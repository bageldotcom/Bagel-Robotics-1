import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('handoff', Path(__file__).with_name('watch_handoff.py'))
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)


def setup_parent(root, update=1251):
    d = root / 'mask/checkpoints' / f'update_{update:06d}'
    d.mkdir(parents=True)
    (d.parent / 'latest').write_text(d.name)
    h.atomic(d / 'COMPLETE.json', dict(update=update, stage='mask', identity='same'))
    h.atomic(root / 'mask/status.json', dict(update=update, status='checkpointed_interruption', identity='same'))
    (root / 'mask/metrics.jsonl').write_text(json.dumps(dict(update=update)) + '\n')


def test_completed_exact_or_next_update(tmp_path):
    setup_parent(tmp_path)
    assert h.completed_parent(tmp_path, 1250, 'same')[0] == 1251


def test_no_rewind_reported_update(tmp_path):
    setup_parent(tmp_path)
    (tmp_path / 'mask/metrics.jsonl').write_text('{"update":1252}\n')
    with pytest.raises(ValueError, match='rewind'):
        h.completed_parent(tmp_path, 1250, 'same')


def test_reject_early_checkpoint(tmp_path):
    setup_parent(tmp_path, 1249)
    with pytest.raises(ValueError, match='outside'):
        h.completed_parent(tmp_path, 1250, 'same')


def test_reject_stale_graceful_status(tmp_path):
    setup_parent(tmp_path)
    h.atomic(tmp_path / 'mask/status.json', dict(update=556, status='checkpointed_interruption', identity='same'))
    with pytest.raises(ValueError, match='graceful'):
        h.completed_parent(tmp_path, 1250, 'same')


def test_ignore_partial_metric(tmp_path):
    setup_parent(tmp_path)
    with (tmp_path / 'mask/metrics.jsonl').open('a') as f:
        f.write('{"update":125')
    assert h.latest_metric(tmp_path)['update'] == 1251


def test_wrong_checkpoint_identity(tmp_path):
    setup_parent(tmp_path)
    with pytest.raises(ValueError, match='identity'):
        h.completed_parent(tmp_path, 1250, 'wrong')


def make_metrics(root, indices, multiplier=1):
    root.mkdir(exist_ok=True)
    (root / 'mask').mkdir(exist_ok=True)
    rows = [dict(update=u, size_weight=.01 * (1 + (u - 1251) / 250) * multiplier,
                 teacher_video_weight=.05, late_kv_weight=.05, total=.3, gt_action=.1,
                 seconds_per_update=52) for u in indices]
    (root / 'mask/metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))


def test_verify_three_consecutive_updates(tmp_path):
    make_metrics(tmp_path, [1252, 1253, 1254])
    assert len(h.check_first_updates(tmp_path, 1251, .01)) == 3


def test_wait_for_three_updates(tmp_path):
    make_metrics(tmp_path, [1252, 1253])
    assert h.check_first_updates(tmp_path, 1251, .01) is None


def test_wrong_pressure_rejected(tmp_path):
    make_metrics(tmp_path, [1252, 1253, 1254], multiplier=2)
    with pytest.raises(ValueError, match='coefficient'):
        h.check_first_updates(tmp_path, 1251, .01)


def test_skipped_update_rejected(tmp_path):
    make_metrics(tmp_path, [1252, 1254, 1255])
    with pytest.raises(ValueError, match='skipped'):
        h.check_first_updates(tmp_path, 1251, .01)


def test_refuses_to_gate_running_job(monkeypatch):
    monkeypatch.setattr(h, 'job', lambda _: dict(JobState='RUNNING'))
    instance = object.__new__(h.Handoff)
    with pytest.raises(RuntimeError, match='pending'):
        instance.hold('50255')


def test_atomic_write(tmp_path):
    h.atomic(tmp_path / 'test.json', dict(complete=True))
    assert h.read(tmp_path / 'test.json') == dict(complete=True)
    assert list(tmp_path.glob('*.tmp')) == []


def recovery_fixture(tmp_path, monkeypatch, child_state, update):
    root = tmp_path / 'child'; root.mkdir()
    selection = tmp_path / 'selection.json'
    monkeypatch.setattr(h, 'ROOT', root)
    monkeypatch.setattr(h, 'SELECTION', selection)
    monkeypatch.setattr(h, 'job', lambda jid: dict(JobState=child_state if jid == h.NEXT_JOB else 'PENDING'))
    monkeypatch.setattr(h, 'latest_metric', lambda _: None if update is None else dict(update=update))
    commands = []
    monkeypatch.setattr(h, 'run', lambda command: (commands.append(command) or '59999'))
    monkeypatch.setattr(h, 'waiting_for_gate', lambda _: False)
    h.atomic(root / 'control/launch_gate.json', dict(managed_jobs=[h.NEXT_JOB, h.BACKUP_JOB], releases={}))
    instance = object.__new__(h.Handoff)
    instance.held = {h.BACKUP_JOB}
    if child_state == 'PENDING': instance.held.add(h.NEXT_JOB)
    instance.switched = True; instance.stopped_update = 1251
    instance.original = dict(root=str(h.PARENT), selected='baseline')
    events = []
    instance.event = lambda phase, **details: events.append((phase, details))
    h.atomic(selection, dict(root=str(root), selected='pressure2x'))
    return instance, selection, commands, events


def test_fallback_before_new_updates(tmp_path, monkeypatch):
    instance, selection, commands, events = recovery_fixture(tmp_path, monkeypatch, 'FAILED', 1251)
    instance.recover(RuntimeError('startup rejected'))
    assert h.read(selection) == instance.original
    assert commands == [['sbatch', '--parsable', str(h.REVIEW / 'slurm/reviewed_train.sbatch')]]
    assert events[-1][1]['fallback_job'] == '59999'
    assert instance.held == set()
    assert events[-1][0] == 'original_route_restored'


def test_no_fallback_after_new_progress(tmp_path, monkeypatch):
    instance, selection, commands, events = recovery_fixture(tmp_path, monkeypatch, 'FAILED', 1252)
    instance.recover(RuntimeError('failed after progress'))
    assert h.read(selection)['selected'] == 'pressure2x'
    assert commands == []
    assert events[-1][0] == 'needs_review_training_or_progress_preserved'


def test_no_routing_change_during_live_continuation(tmp_path, monkeypatch):
    instance, selection, commands, events = recovery_fixture(tmp_path, monkeypatch, 'RUNNING', 1251)
    instance.recover(RuntimeError('watcher timed out'))
    assert h.read(selection)['selected'] == 'pressure2x'
    assert commands == []


def test_preparation_failure_releases_original_chain(tmp_path, monkeypatch):
    instance, selection, commands, events = recovery_fixture(tmp_path, monkeypatch, 'PENDING', 1251)
    instance.recover(RuntimeError('CPU binding rejected'))
    assert h.read(selection) == instance.original
    assert instance.held == set()
    assert not any('update' in command for command in commands)


def test_preparation_failure_releases_waiting_launcher(tmp_path, monkeypatch):
    instance, selection, commands, events = recovery_fixture(tmp_path, monkeypatch, 'RUNNING', 1251)
    instance.held.add(h.NEXT_JOB)
    monkeypatch.setattr(h, 'waiting_for_gate', lambda _: True)
    instance.recover(RuntimeError('routing check failed while launcher is waiting'))
    assert h.read(selection) == instance.original
    assert instance.held == set()
    assert commands == []


def test_gate_release_never_changes_slurm_resources(tmp_path, monkeypatch):
    instance, selection, commands, events = recovery_fixture(tmp_path, monkeypatch, 'PENDING', 1251)
    instance.release(h.NEXT_JOB)
    gate = h.read(h.ROOT / 'control/launch_gate.json')
    assert gate['releases'][h.NEXT_JOB]['root'] == str(h.ROOT)
    assert commands == []


def test_launcher_wait_and_release_with_real_process(tmp_path):
    import importlib.util
    import os
    import subprocess
    import sys
    import time
    helper = Path(__file__).with_name('launcher_gate.py')
    spec = importlib.util.spec_from_file_location('gate_check', helper)
    gate_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate_module)
    root = tmp_path / 'child'; review = tmp_path / 'review'
    script = review / 'control/prepare_resume.py'; script.parent.mkdir(parents=True)
    h.atomic(root / 'control/launch_gate.json', dict(managed_jobs=['test-wait'], releases={}))
    h.atomic(review / 'control/production_selection.json', dict(root=str(root)))
    script.write_text('import importlib.util\nfrom pathlib import Path\n'
        + f'spec=importlib.util.spec_from_file_location("gate",{str(helper)!r})\n'
        + 'g=importlib.util.module_from_spec(spec); spec.loader.exec_module(g)\n'
        + f'g.ROOT=Path({str(root)!r}); g.REVIEW=Path({str(review)!r})\ng.main()\n')
    process = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=dict(os.environ, SLURM_JOB_ID='test-wait', CUDA_VISIBLE_DEVICES=''))
    try:
        deadline = time.monotonic()+10
        while time.monotonic()<deadline and not gate_module.is_waiting_job('test-wait',root,review):
            time.sleep(.02)
        assert gate_module.is_waiting_job('test-wait',root,review)
        h.atomic(root / 'control/launch_gate.json', dict(managed_jobs=['test-wait'],
                 releases={'test-wait': dict(root=str(root))}))
        stdout,stderr=process.communicate(timeout=10)
        assert process.returncode==0,stderr.decode()
        assert b'route released' in stdout
        assert not gate_module.is_waiting_job('test-wait',root,review)
    finally:
        if process.poll() is None:
            process.terminate(); process.wait(timeout=5)
