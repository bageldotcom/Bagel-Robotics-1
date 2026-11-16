import json
from pathlib import Path
import numpy as np
import torch
from vpp2.full_mask.curation import history_indices
from vpp2.full_mask.curation_v3_quality import balanced_tasks, run


def test_quality_task_remainders_rotate():
    rng = np.random.default_rng(100008)
    order = np.concatenate([balanced_tasks(i, 256, 35, rng) for i in range(1, 36)])
    assert np.array_equal(np.bincount(order), np.full(35, 256))


def test_quality_pipeline_excludes_audit_and_rejects_stale_model(tmp_path, monkeypatch):
    # Exercise fitting, monitoring, neighbour scores, fallbacks and resume identities.
    # Audit has no content on disk: any attempt to inspect it fails this test.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    for name in ('candidates', 'visual'):
        (tmp_path / name).mkdir()
    rng = np.random.default_rng(3)
    episodes = []
    for ep in range(14):
        role = 'fit' if ep % 7 < 5 else 'monitor'
        e = dict(episode=ep, role=role, task=f'task{ep // 7}')
        episodes.append(e)
        starts = np.arange(0, 256, 32)
        state = rng.normal(size=(len(starts), 16)).astype(np.float32)
        state[:, 3:7] = [0, 0, 0, 1]
        state[:, 11:15] = [0, 0, 0, 1]
        frames = np.array(sorted({f for t in starts for f in history_indices(t)}))
        np.savez(tmp_path / 'candidates' / f'{ep:05d}.npz',
                 starts=starts, state=state, motion=np.zeros((8, 28)),
                 action=rng.normal(size=(8, 144)).astype(np.float32),
                 score_indices=np.arange(8))
        np.savez(tmp_path / 'visual' / f'{ep:05d}.npz', frames=frames,
                 features=rng.normal(size=(len(frames), 288)).astype(np.float32))
    episodes.append(dict(episode=999, role='audit', task='task0'))
    (tmp_path / 'inventory.json').write_text(json.dumps(dict(identity='test', episodes=episodes)))
    (tmp_path / 'visual_features_receipt.json').write_text(json.dumps(
        dict(complete=True, inventory_identity='test', identity='vision')))
    config = dict(seed=100008, curation=dict(output=str(tmp_path), quality=dict(
        max_updates=4, check_every=2, patience=1, learning_rate=1e-3,
        obs_latent=8, action_latent=4, neighbors=[5, 6, 7], repeats=2,
        minimum_rank_correlation=.8)))
    receipt = run(config)
    assert receipt['passed']
    assert receipt['audit_episodes_used'] == 0
    assert len(receipt['episodes']) == 10
    assert not (tmp_path / 'summaries' / '00999.npz').exists()
    assert run(config) == receipt
    with (tmp_path / 'quality_models.pt').open('ab') as f:
        f.write(b'changed')
    import pytest
    with pytest.raises(ValueError, match='Stale quality receipt'):
        run(config)
