"""Task-balanced visual screening, distinct from dense critical-interval labels.

The 280-episode annotation study is an additional audit. A screen attests only
to the images actually inspected; it never manufactures interval boundaries,
memory observability, a success label, or a dataset-wide quality guarantee.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from vpp2.full_mask.panel import digest, file_hash, immutable
from .contracts import checked_identity, read_json, load_annotations

SCHEMA = 'robodojo-staged-visual-screen-v1'
POLICY = dict(
    version=1, initial_fit_episodes_per_task=1,
    choice='Second shortest of the four metadata-selected fitting episodes; no policy scores.',
    include_existing_verified_fit_episodes=True,
    expansion='One additional fitting episode per task with a missing phase or data defect; further flagged episodes expand again.',
    minimum_dense_annotations_for_training=0,
    full_280_annotation_is_launch_gate=False,
    monitor_labels_used_for_selection=False,
    unknown_memory_cues_are_not_verified=True,
    automatic_near_duplicate_deletion=False,
)


def select_initial(manifest):
    result = []
    for task in manifest['tasks']:
        episodes = sorted((e for e in manifest['episodes'] if e['task'] == task and e['role'] == 'fit'),
                          key=lambda e: (e['length'], e['episode']))
        if len(episodes) != 4:
            raise ValueError('Staged screen expects four preselected fitting episodes per task')
        result.append(episodes[1]['episode'])
    return result


def prepare(config):
    root = Path(config['root'])
    manifest = checked_identity(read_json(root/'audit_manifest.json'))
    annotations, _ = load_annotations(config, manifest)
    initial = select_initial(manifest)
    existing = sorted(ep for ep, a in annotations.items() if a['role'] == 'fit')
    value = dict(schema=SCHEMA, policy=POLICY, manifest_identity=manifest['identity'],
                 base_panel_identity=manifest['base_panel_identity'],
                 initial_episodes=initial, existing_dense_fit_episodes=existing,
                 required_episodes=sorted(set(initial) | set(existing)),
                 tasks=manifest['tasks'], policy_predictions_used=False)
    value['identity'] = digest(value)
    # An existing manifest remains the fixed preregistered selection even when
    # additional dense annotations arrive later.
    destination = root/'staged_review/manifest.json'
    if destination.exists():
        old = checked_identity(read_json(destination))
        if old['manifest_identity'] != manifest['identity'] or old['policy'] != POLICY:
            raise ValueError('Staged review policy/source changed; version the protocol')
        return old
    immutable(destination, value)
    return value


def validate_record(config, manifest, stage, record):
    if record.get('schema') != SCHEMA or record.get('stage_identity') != stage['identity']:
        raise ValueError('Screen belongs to another protocol')
    episodes = {e['episode']: e for e in manifest['episodes']}
    ep = episodes.get(record.get('episode'))
    if ep is None or ep['role'] != 'fit':
        raise ValueError('Only fitting episodes may influence selection')
    packet_root = Path(config['root'])/'packets'/f"{ep['episode']:05d}"
    packet = checked_identity(read_json(packet_root/'packet.json'))
    if record.get('packet_identity') != packet['identity'] or record.get('task') != ep['task']:
        raise ValueError('Screen episode/packet mismatch')
    if record.get('status') != 'screened' or not str(record.get('reviewer', '')).strip():
        raise ValueError('A real named image inspection is required')
    if record.get('full_coarse_timeline_inspected') is not True:
        raise ValueError('Inspect every coarse timeline page')
    expected = {p['file']: p['sha256'] for p in packet['coarse_pages']}
    if record.get('inspected_pages') != expected:
        raise ValueError('Missing or stale image evidence')
    for name, sha in expected.items():
        if file_hash(packet_root/name) != sha:
            raise ValueError('Screen image bytes changed')
    if not str(record.get('observations', '')).strip():
        raise ValueError('Describe the observed sequence')
    if record.get('critical_interval_boundaries_verified') is not False:
        raise ValueError('A coarse screen cannot certify interval boundaries')
    shown = {t for p in packet['coarse_pages'] for t in p['frames']}
    anchors = record.get('manipulation_anchor_frames')
    if not isinstance(anchors, list) or not anchors or len(set(anchors)) != len(anchors):
        raise ValueError('Record distinct observed manipulation anchors')
    if any(type(t) is not int or t not in shown for t in anchors):
        raise ValueError('Anchors must be original frames actually shown in the inspected pages')
    if record.get('memory_cue_status') not in ('not_applicable', 'uncertain'):
        raise ValueError('Coarse screening must not claim verified memory cues')
    if ep['suite'] == 'memory' and record['memory_cue_status'] != 'uncertain':
        raise ValueError('Memory cue observability needs a separate dense annotation')
    findings = record.get('findings')
    if not isinstance(findings, list):
        raise ValueError('Findings must be explicit, including an empty list')
    seen = set()
    for finding in findings:
        if not finding.get('id') or finding['id'] in seen:
            raise ValueError('Finding identities must be unique')
        seen.add(finding['id'])
        if finding.get('kind') not in ('missing_phase', 'data_defect', 'uncertain_interpretation'):
            raise ValueError('Unknown finding type')
        if not str(finding.get('evidence', '')).strip():
            raise ValueError('Findings require evidence')
        frames = finding.get('frames', [])
        if any(type(t) is not int or t not in shown for t in frames):
            raise ValueError('Finding frames must be inspected')
        if finding['kind'] == 'missing_phase' and not frames:
            raise ValueError('A missing phase needs observed frames to protect')
        if finding['kind'] == 'missing_phase' and not set(frames) <= set(anchors):
            raise ValueError('Missing-phase frames must become protected anchors')
    return record


def import_record(config, path):
    root = Path(config['root']); stage = prepare(config)
    manifest = checked_identity(read_json(root/'audit_manifest.json'))
    record = validate_record(config, manifest, stage, read_json(path))
    rid = digest(record)
    destination=root/'staged_review/records'/f"{record['episode']:05d}.json"
    if destination.exists():
        previous=read_json(destination)
        if previous!=record and record.get('supersedes_record_identity')!=digest(previous):
            raise ValueError('Review revision must explicitly identify the previous record')
        immutable(root/'staged_review/history'/f"{record['episode']:05d}_{digest(previous)[:16]}.json",previous)
    immutable(root/'staged_review/history'/f"{record['episode']:05d}_{rid[:16]}.json",record)
    immutable(root/'staged_review/receipts'/f"{record['episode']:05d}_{rid[:16]}.json",
              dict(record_identity=rid, stage_identity=stage['identity'], image_hashes_checked=True))
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp=destination.with_suffix('.partial');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(destination)
    return record


def load_records(config, manifest, stage):
    root = Path(config['root']); result = {}
    for path in sorted((root/'staged_review/records').glob('*.json')):
        record = validate_record(config, manifest, stage, read_json(path)); ep = record['episode']
        if ep in result:
            raise ValueError('Duplicate episode screen')
        rid = digest(record)
        receipt = read_json(root/'staged_review/receipts'/f'{ep:05d}_{rid[:16]}.json')
        if receipt != dict(record_identity=rid, stage_identity=stage['identity'], image_hashes_checked=True):
            raise ValueError('Screen import receipt differs')
        result[ep] = record
    return result


def assess(config, panel=None):
    root = Path(config['root']); stage = prepare(config)
    manifest = checked_identity(read_json(root/'audit_manifest.json'))
    records = load_records(config, manifest, stage)
    required = set(stage['required_episodes']); expansions = {}
    for task in stage['tasks']:
        flagged = {ep for ep, r in records.items() if r['task'] == task and
                   any(f['kind'] in ('missing_phase', 'data_defect') for f in r['findings'])}
        remaining = sorted((e for e in manifest['episodes'] if e['task'] == task and
                            e['role'] == 'fit' and e['episode'] not in required),
                           key=lambda e: (e['length'], e['episode']))
        added = [e['episode'] for e in remaining[:len(flagged)]]
        required.update(added)
        if flagged:
            expansions[task] = dict(flagged_episodes=sorted(flagged), additional_episodes=added)
    missing = sorted(required-set(records)); defects = []
    anchors = {}; record_ids = {}
    for ep, r in records.items():
        anchors[ep] = dict(role='fit', frames=r['manipulation_anchor_frames'])
        record_ids[ep] = digest(r)
        defects.extend(dict(episode=ep, **f) for f in r['findings'] if f['kind'] == 'data_defect')
    unresolved_anchors = []
    if panel is not None:
        if panel.get('identity') != stage['base_panel_identity'] and panel.get('parent_panel_identity') != stage['base_panel_identity']:
            raise ValueError('Screen applied to unrelated data')
        by_episode = defaultdict(list)
        for row in panel['windows']:
            if row['role'] == 'selection_initial':
                by_episode[row['episode']].append(row['action_interval'])
        for ep, a in anchors.items():
            for t in a['frames']:
                if not any(start <= t < end for start, end in by_episode[ep]):
                    unresolved_anchors.append([ep, t])
    result = dict(schema=SCHEMA, stage_identity=stage['identity'],
        base_panel_identity=stage['base_panel_identity'],
        panel_identity=panel['identity'] if panel is not None else None,
        required_episodes=sorted(required), screened_episodes=sorted(records), missing_episodes=missing,
        expansions=expansions, unresolved_data_defects=defects,
        unresolved_manipulation_anchors=unresolved_anchors, record_identities=record_ids,
        selection_ready=not missing and not defects,
        passed=panel is not None and not missing and not defects and not unresolved_anchors,
        all_280_annotations_required=False, training_ready=False,
        claim='Task-balanced coarse image screen only; unannotated phases/cues remain unknown.',
        remaining_training_requirements=['Current-panel CPU integrity', 'Native GPU equivalence',
                                         'Finite action-sensitivity probes and suite review'])
    result['identity'] = digest(result)
    return result, anchors


def write_status(config, panel=None):
    result, _ = assess(config, panel)
    root = Path(config['root'])/'staged_review'
    immutable(root/'reports'/f"{result['identity'][:16]}.json", result)
    (root/'status.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True)
    p.add_argument('--import-record'); p.add_argument('--panel')
    a = p.parse_args(); config = read_json(a.config); prepare(config)
    if a.import_record:
        import_record(config, a.import_record)
    panel = checked_identity(read_json(a.panel)) if a.panel else None
    print(json.dumps(write_status(config, panel), indent=2))


if __name__ == '__main__':
    main()
