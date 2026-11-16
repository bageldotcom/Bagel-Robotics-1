import pytest
import torch
from vpp2.full_mask.curation_v3_probe import sample_rows, similarity


def test_probe_samples_independent_episodes_without_audit_or_role_mix():
    rows = [dict(task='t', role=role, episode=100*j+ep, frame=t, window_id=f'{role}-{ep}-{t}')
            for j, role in enumerate(('selection_initial', 'monitor', 'audit'))
            for ep in range(8) for t in (0, 32)]
    panel = {'windows': rows}
    a = sample_rows(panel, 't', 'selection_initial', 8, 100008)
    b = sample_rows(panel, 't', 'monitor', 8, 100008)
    assert len({r['episode'] for r in a}) == 8
    assert len({r['episode'] for r in b}) == 8
    assert not {r['episode'] for r in a} & {r['episode'] for r in b}
    assert all(r['role'] != 'audit' for r in a+b)
    assert sample_rows(panel, 't', 'monitor', 8, 100008) == b
    with pytest.raises(ValueError):
        sample_rows(panel, 't', 'monitor', 9, 100008)


def test_sensitivity_comparison_does_not_call_zero_gradients_evidence():
    a = torch.zeros(100)
    assert similarity(a, a)['cosine'] is None
    assert similarity(a, a)['top5_jaccard'] is None
    a[1] = 1
    assert similarity(a, a)['cosine'] == 1
