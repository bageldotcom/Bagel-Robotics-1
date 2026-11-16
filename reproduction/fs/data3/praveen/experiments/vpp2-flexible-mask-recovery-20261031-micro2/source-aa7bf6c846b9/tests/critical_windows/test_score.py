import json
import numpy as np
import pytest
from vpp2.full_mask.panel import digest
from vpp2.critical_windows.score import score


def prediction_fixture(tmp_path):
    ep = dict(episode=1, task='task', suite='precision', role='monitor', length=25)
    manifest = dict(episodes=[ep])
    manifest['identity'] = digest(manifest)
    (tmp_path / 'audit_manifest.json').write_text(json.dumps(manifest))
    predictions = tmp_path / 'predictions'
    predictions.mkdir()
    identity = dict(identity='fixed', manifest_identity=manifest['identity'], role='monitor', variant='native')
    (predictions / 'identity.json').write_text(json.dumps(identity))
    def write(start, valid):
        values = np.zeros((32, 16), dtype=np.float32)
        values[:, 3] = 1
        values[:, 11] = 1
        np.savez_compressed(predictions / f'00001_{start:05d}.npz',
                            identity='fixed', episode=1, frame=start, task='task', suite='precision',
                            valid_steps=valid, normalized=values, target_normalized=values,
                            physical=values, target_physical=values)
    return dict(root=str(tmp_path)), predictions, write


def test_scoring_requires_complete_queries_and_does_not_invent_critical_labels(tmp_path):
    config, predictions, write = prediction_fixture(tmp_path)
    write(0, 25)
    with pytest.raises(ValueError, match='1 of 2'):
        score(config, predictions)
    _, partial = score(config, predictions, allow_partial=True)
    assert partial['predictions_complete'] is False
    assert partial['missing_queries'] == [[1, 24]]
    write(24, 1)
    _, complete = score(config, predictions)
    assert complete['predictions_complete'] is True
    assert complete['verified_evaluated_episodes'] == 0
    assert complete['per_episode'][0]['steps'] == 25
    assert {row['group'] for row in complete['per_episode']} == {'all_executed'}
    assert complete['policy_retention_established'] is False
    assert complete['metric_source_identity']


def test_scoring_rejects_bad_tail_metadata(tmp_path):
    config, predictions, write = prediction_fixture(tmp_path)
    write(0, 25)
    write(24, 32)
    with pytest.raises(ValueError, match='metadata mismatch'):
        score(config, predictions)
