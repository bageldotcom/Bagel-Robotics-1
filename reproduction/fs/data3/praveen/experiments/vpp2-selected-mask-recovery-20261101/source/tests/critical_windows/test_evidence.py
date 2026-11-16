import copy
import json
import pytest
from vpp2.full_mask.panel import digest, file_hash
from vpp2.critical_windows.contracts import load_annotations, validate_evidence_files, validate_annotation
from vpp2.critical_windows.coverage import import_annotation, coverage_for_episode
from test_contracts_metrics import annotation_fixture, interval


def evidence_fixture(tmp_path):
    manifest, packet, annotation = annotation_fixture()
    root = tmp_path / 'packets' / '00001'
    root.mkdir(parents=True)
    (root / 'coarse.jpg').write_bytes(b'coarse review fixture')
    (root / 'dense.jpg').write_bytes(b'dense review fixture')
    packet['coarse_pages'][0]['sha256'] = file_hash(root / 'coarse.jpg')
    packet['identity'] = digest({k: v for k, v in packet.items() if k != 'identity'})
    annotation['packet_identity'] = packet['identity']
    annotation['inspected_dense_pages'][0]['sha256'] = file_hash(root / 'dense.jpg')
    (root / 'packet.json').write_text(json.dumps(packet))
    dense = dict(episode=1, manifest_identity=manifest['identity'], pages=[
        dict(file='dense.jpg', sha256=file_hash(root / 'dense.jpg'), frames=[10])])
    (root / 'dense_fixture.json').write_text(json.dumps(dense))
    source = tmp_path / 'annotation.json'
    source.write_text(json.dumps(annotation))
    return manifest, packet, annotation, source


def test_verified_annotations_require_matching_import_and_unchanged_evidence(tmp_path):
    manifest, packet, annotation, source = evidence_fixture(tmp_path)
    config = dict(root=str(tmp_path))
    (tmp_path / 'annotations').mkdir()
    destination = tmp_path / 'annotations' / '00001.json'
    destination.write_text(json.dumps(annotation))
    with pytest.raises(ValueError, match='import receipt'):
        load_annotations(config, manifest)
    import_annotation(config, manifest, source)
    assert load_annotations(config, manifest)[0][1] == annotation
    (tmp_path / 'packets' / '00001' / 'dense.jpg').write_bytes(b'altered')
    with pytest.raises(ValueError, match='evidence changed'):
        load_annotations(config, manifest)


def test_claimed_evidence_must_exist_in_inspected_dense_pages(tmp_path):
    manifest, packet, annotation, _ = evidence_fixture(tmp_path)
    annotation['intervals'][0]['evidence_frames'] = [10, 11]
    with pytest.raises(ValueError, match='absent'):
        validate_evidence_files(tmp_path, annotation, packet, manifest)


def test_boundary_uncertainty_does_not_hide_full_interval_coverage():
    ep = dict(length=50, episode=1, task='task', suite='precision', role='fit')
    phase = dict(interval(8, 44), boundary_uncertainty_frames=2)
    report = coverage_for_episode(ep, [dict(action_interval=[10, 42], frame=10, window_id='w')],
                                  dict(intervals=[phase]))
    result = report['intervals'][0]
    assert result['coverage_fraction'] == 32 / 36
    assert result['inner_interval_coverage_fraction'] == 1.0
    assert result['fully_covered'] is False
    manifest, packet, annotation = annotation_fixture()
    for invalid in (-1, 1.5, True, 100):
        value = copy.deepcopy(annotation)
        value['intervals'][0]['boundary_uncertainty_frames'] = invalid
        with pytest.raises(ValueError, match='Boundary uncertainty'):
            validate_annotation(manifest, value, packet)
