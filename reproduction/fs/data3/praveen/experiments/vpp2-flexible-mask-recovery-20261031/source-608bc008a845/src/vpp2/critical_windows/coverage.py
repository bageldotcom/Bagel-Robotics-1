"""Measure critical-phase coverage without inventing labels for other episodes."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import numpy as np
from vpp2.full_mask.panel import digest, immutable, file_hash
from .contracts import (read_json, checked_identity, load_annotations, validate_annotation,
                        critical_mask, cue_visible, validate_evidence_files)


def coverage_for_episode(episode, selected, annotation):
    length = episode['length']; union = np.zeros(length, dtype=bool)
    for row in selected:
        a, b = row['action_interval']; union[a:b] = True
    all_critical = critical_mask(0, length, length, annotation['intervals'])
    intervals = []
    for item in annotation['intervals']:
        a, b = item['frames']; count = int(union[a:b].sum())
        uncertainty = int(item.get('boundary_uncertainty_frames', 0))
        inner_a, inner_b = a + uncertainty, b - uncertainty
        inner_total = max(0, inner_b-inner_a)
        inner_covered = int(union[inner_a:inner_b].sum()) if inner_total else 0
        overlapping = [r for r in selected if r['frame'] < b and r['frame'] + 32 > a]
        cues = [cue_visible(r['frame'], item) for r in overlapping]
        intervals.append(dict(id=item['id'], kind=item['kind'], frames=item['frames'],
                              covered_steps=count, total_steps=b-a, coverage_fraction=count/(b-a),
                              fully_covered=count == b-a,
                              boundary_uncertainty_frames=uncertainty,
                              inner_interval_coverage_fraction=inner_covered/inner_total if inner_total else None,
                              intersecting_windows=[r['window_id'] for r in overlapping],
                              native_history_cue_visible=any(x is True for x in cues) if any(x is not None for x in cues) else None))
    total = int(all_critical.sum()); covered = int((all_critical & union).sum())
    return dict(episode=episode['episode'], task=episode['task'], suite=episode['suite'], role=episode['role'],
                selected_windows=len(selected), critical_steps=total, covered_critical_steps=covered,
                critical_coverage_fraction=covered/total if total else None, intervals=intervals)


def audit(config, manifest):
    annotations, reviews = load_annotations(config, manifest)
    panel = checked_identity(read_json(Path(config['base_data']) / 'panel.json'))
    if panel['identity'] != manifest['base_panel_identity']:
        raise ValueError('Base subset changed')
    by_ep = defaultdict(list)
    for row in panel['windows']:
        if row['role'] in ('selection_initial', 'monitor'):
            by_ep[row['episode']].append(row)
    result = []
    for e in manifest['episodes']:
        if e['episode'] in annotations:
            result.append(coverage_for_episode(e, by_ep[e['episode']], annotations[e['episode']]))
    by_task = {}
    for task in manifest['tasks']:
        entries = [r for r in result if r['task'] == task]
        by_task[task] = dict(verified_episodes=len(entries),
                            expected_episodes=sum(e['task'] == task for e in manifest['episodes']),
                            intervals=sum(len(e['intervals']) for e in entries),
                            fully_covered_intervals=sum(x['fully_covered'] for e in entries for x in e['intervals']))
    receipt = dict(manifest_identity=manifest['identity'], base_panel_identity=panel['identity'],
                   verified_episodes=len(annotations), expected_episodes=len(manifest['episodes']),
                   annotation_complete=len(annotations) == len(manifest['episodes']),
                   reviews=reviews, by_task=by_task, episodes=result,
                   unknown_is_not_zero=True, training_ready=False,
                   note='Coverage measures the annotated development sample only; it cannot certify unreviewed episodes.')
    identity = digest(receipt); receipt['identity'] = identity
    out = Path(config['root']) / 'reports'; out.mkdir(exist_ok=True)
    immutable(out / f'coverage_{identity[:12]}.json', receipt)
    # Pointer only; historical reports stay immutable.
    (out / 'latest_coverage.json').write_text(json.dumps({'report': str(out / f'coverage_{identity[:12]}.json'), 'identity': identity}, indent=2)+'\n')
    return receipt


def import_annotation(config, manifest, path):
    value = read_json(path); ep = value['episode']
    packet = checked_identity(read_json(Path(config['root']) / 'packets' / f'{ep:05d}' / 'packet.json'))
    validate_annotation(manifest, value, packet)
    validate_evidence_files(config['root'], value, packet, manifest)
    out = Path(config['root']) / 'annotations'; out.mkdir(exist_ok=True)
    content_id = digest(value)
    immutable(out/'receipts'/f'{ep:05d}_{content_id[:16]}.json', dict(annotation_identity=content_id,manifest_identity=manifest['identity'],evidence_files_checked=True,reviewer_attestation=value.get('reviewer'),status=value['status']))
    immutable(out / 'history' / f'{ep:05d}_{content_id[:12]}.json', value)
    destination = out / f'{ep:05d}.json'
    if destination.exists() and read_json(destination).get('status') == 'verified' and value.get('status') != 'verified':
        raise ValueError('Do not silently replace a verified review with a draft')
    tmp = destination.with_suffix('.partial'); tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(destination)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--import-annotation')
    a = p.parse_args(); c = read_json(a.config); m = checked_identity(read_json(Path(c['root']) / 'audit_manifest.json'))
    if a.import_annotation:
        import_annotation(c, m, a.import_annotation)
    report = audit(c, m)
    print(json.dumps({k: report[k] for k in ('verified_episodes', 'expected_episodes', 'annotation_complete', 'training_ready')}))


if __name__ == '__main__':
    main()
