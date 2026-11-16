"""Versioned, fail-closed contracts for model-blind episode annotation."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from vpp2.full_mask.panel import digest, file_hash, immutable, seed_for

SCHEMA = 'robodojo-critical-intervals-v1'
KINDS = {'grasp', 'precision_alignment', 'contact_manipulation', 'controlled_release',
         'bimanual_coordination', 'memory_decision', 'required_subgoal', 'recovery'}


def read_json(path):
    return json.loads(Path(path).read_text())


def checked_identity(value):
    if value.get('identity') != digest({k: v for k, v in value.items() if k != 'identity'}):
        raise ValueError('Artifact identity mismatch')
    return value


def make_manifest(config):
    """Four length-stratified random episodes per task and development role.

    The choice uses metadata only, not policy predictions, validation scores or
    visual criticality. Sealed audit and official validation episodes are banned.
    """
    inventory_path = Path(config['base_data']) / 'inventory.json'
    inv = checked_identity(read_json(inventory_path))
    panel = checked_identity(read_json(Path(config['base_data']) / 'panel.json'))
    if panel['inventory_identity'] != inv['identity']:
        raise ValueError('Inventory and panel differ')
    per_role = config['episodes_per_task_per_role']
    episodes = []
    for task in inv['tasks']:
        for role in ('fit', 'monitor'):
            candidates = sorted((e for e in inv['episodes'] if e['task'] == task and e['role'] == role),
                                key=lambda e: (e['length'], e['episode']))
            if len(candidates) < per_role:
                raise ValueError(f'Insufficient {role} episodes for {task}')
            rng = np.random.default_rng(seed_for(config['seed'], 'critical-episode-panel', task, role))
            for bucket in np.array_split(np.arange(len(candidates)), per_role):
                episodes.append(candidates[int(rng.choice(bucket))])
    banned = set(inv['official_validation_episodes']) | {e['episode'] for e in inv['episodes'] if e['role'] == 'audit'}
    if any(e['episode'] in banned for e in episodes):
        raise ValueError('Sealed/official audit episode entered the development panel')
    if len({e['episode'] for e in episodes}) != len(episodes):
        raise ValueError('Episode repeated across roles')
    result = dict(schema=SCHEMA, seed=config['seed'], base_panel_identity=panel['identity'],
                  inventory_identity=inv['identity'], inventory_sha256=file_hash(inventory_path),
                  source_identity=inv['source_identity'], source=inv['source'],
                  episodes=episodes, tasks=inv['tasks'], counts=dict(Counter(e['role'] for e in episodes)),
                  selection='Metadata-only length strata, reproducible random episode within each stratum.',
                  timestamp_unit='original_decoded_frame_index',
                  policy_predictions_used=False, sealed_audit_opened=False)
    result['identity'] = digest(result)
    immutable(Path(config['root']) / 'audit_manifest.json', result)
    return result


def annotation_template(manifest, episode, packet_identity):
    return dict(schema=SCHEMA, manifest_identity=manifest['identity'], packet_identity=packet_identity,
                episode=episode['episode'], task=episode['task'], role=episode['role'],
                length=episode['length'], status='unreviewed', reviewer=None,
                full_timeline_inspected=False, inspected_coarse_pages=[], inspected_dense_pages=[], intervals=[],
                no_critical_interval_reason=None, annotation_notes='',
                restriction='Source images only; no checkpoint predictions or rollout scores.')


def frame_interval(pair, length):
    if not isinstance(pair, list) or len(pair) != 2 or any(type(x) is not int for x in pair):
        raise ValueError('Intervals require two integer ORIGINAL frame indices, [start,end)')
    a, b = pair
    if not 0 <= a < b <= length:
        raise ValueError('Interval out of bounds or empty')
    return a, b


def validate_annotation(manifest, annotation, packet):
    """Validate provenance and an explicit visual-review attestation, not its truth.

    Metadata validity alone must never be reported as semantic verification.
    Unreviewed/draft annotations are retained but cannot enter metric masks.
    """
    by_id = {e['episode']: e for e in manifest['episodes']}
    ep = by_id.get(annotation.get('episode'))
    if ep is None or ep['role'] not in ('fit', 'monitor'):
        raise ValueError('Unknown or sealed episode')
    for key, expected in [('schema', SCHEMA), ('manifest_identity', manifest['identity']),
                          ('packet_identity', packet['identity']), ('task', ep['task']),
                          ('role', ep['role']), ('length', ep['length'])]:
        if annotation.get(key) != expected:
            raise ValueError(f'Annotation {key} mismatch')
    if packet['episode'] != ep['episode'] or packet['manifest_identity'] != manifest['identity']:
        raise ValueError('Wrong visual packet')
    status = annotation.get('status')
    if status not in ('unreviewed', 'draft', 'verified', 'uncertain'):
        raise ValueError('Unknown annotation status')
    if status != 'verified':
        return dict(usable=False, reason=status, episode=ep['episode'])
    if not isinstance(annotation.get('reviewer'), str) or not annotation['reviewer'].strip():
        raise ValueError('Verified labels require a named visual reviewer')
    if annotation.get('full_timeline_inspected') is not True:
        raise ValueError('Full timeline inspection is required')
    expected_pages = {p['file'] for p in packet['coarse_pages']}
    if set(annotation.get('inspected_coarse_pages', [])) != expected_pages:
        raise ValueError('Full timeline page evidence is incomplete')
    intervals = annotation.get('intervals')
    if not isinstance(intervals, list):
        raise ValueError('Intervals must be a list')
    if not intervals and not str(annotation.get('no_critical_interval_reason') or '').strip():
        raise ValueError('Verified empty annotations require a reason')
    if intervals and not annotation.get('inspected_dense_pages'):
        raise ValueError('Verified interval boundaries require dense visual evidence')
    ids = set()
    for item in intervals:
        a, b = frame_interval(item.get('frames'), ep['length'])
        uncertainty = item.get('boundary_uncertainty_frames', 0)
        if type(uncertainty) is not int or uncertainty < 0 or 2 * uncertainty > b - a:
            raise ValueError('Boundary uncertainty must be a nonnegative integer no greater than half the interval')
        if item.get('kind') not in KINDS or not str(item.get('label', '')).strip():
            raise ValueError('Critical interval needs a known kind and descriptive label')
        if not item.get('id') or item['id'] in ids:
            raise ValueError('Missing or duplicate interval identity')
        ids.add(item['id'])
        evidence = item.get('evidence_frames', [])
        if not evidence or any(type(t) is not int or not 0 <= t < ep['length'] for t in evidence):
            raise ValueError('Invalid visual evidence frames')
        if not any(a <= t < b for t in evidence):
            raise ValueError('Evidence must include the interval interior')
        if not any(abs(t-a)<=8 for t in evidence) or not any(abs(t-(b-1))<=8 for t in evidence):
            raise ValueError('Evidence is missing near an interval boundary')
        if not str(item.get('evidence', '')).strip():
            raise ValueError('Describe the observed interaction; metadata tags are not evidence')
        if item['kind'] == 'memory_decision':
            if item.get('cue_status') not in ('identified', 'unobservable', 'uncertain'):
                raise ValueError('Memory decisions require explicit cue status')
            cues = item.get('cue_intervals', [])
            if item['cue_status'] == 'identified' and (not cues or item.get('cue_requirement') != 'all_intervals'):
                raise ValueError('Identified memory cues need all required observation ranges explicitly specified')
            for cue in cues:
                c, d = frame_interval(cue, ep['length'])
                if c >= b:
                    raise ValueError('A future cue cannot justify an earlier decision')
    return dict(usable=True, episode=ep['episode'], intervals=len(intervals),
                verification='Named visual-review attestation; schema checks alone do not prove semantic accuracy.')


def validate_evidence_files(root, annotation, packet, manifest):
    packet_root = Path(root) / 'packets' / f"{annotation['episode']:05d}"
    for page in packet['coarse_pages']:
        if Path(page['file']).name != page['file']:
            raise ValueError('Visual evidence must stay inside the episode packet')
        if file_hash(packet_root / page['file']) != page['sha256']:
            raise ValueError('Coarse visual evidence changed')
    if annotation.get('status') != 'verified' or not annotation.get('intervals'):
        return
    dense_pages = {}
    for path in packet_root.glob('dense_*.json'):
        dense = read_json(path)
        if dense['episode'] != annotation['episode'] or dense['manifest_identity'] != manifest['identity']:
            raise ValueError('Dense evidence belongs to another episode')
        dense_pages.update({page['file']: page for page in dense['pages']})
    seen_frames = set()
    for page in annotation.get('inspected_dense_pages', []):
        stored = dense_pages.get(page.get('file'))
        if stored is None or page.get('sha256') != stored['sha256'] or Path(stored['file']).name != stored['file']:
            raise ValueError('Missing or altered dense visual evidence')
        if file_hash(packet_root / stored['file']) != stored['sha256']:
            raise ValueError('Dense visual evidence changed')
        seen_frames.update(stored['frames'])
    for interval in annotation['intervals']:
        if not set(interval['evidence_frames']).issubset(seen_frames):
            raise ValueError('Claimed evidence frame is absent from inspected dense pages')


def load_annotations(config, manifest):
    root = Path(config['root']); usable = {}; receipt = []
    for ep in manifest['episodes']:
        path = root / 'annotations' / f"{ep['episode']:05d}.json"
        if not path.exists():
            receipt.append(dict(episode=ep['episode'], usable=False, reason='missing')); continue
        annotation = read_json(path)
        packet = checked_identity(read_json(root / 'packets' / f"{ep['episode']:05d}" / 'packet.json'))
        item = validate_annotation(manifest, annotation, packet)
        item['annotation_sha256'] = file_hash(path); receipt.append(item)
        if item['usable']:
            validate_evidence_files(root, annotation, packet, manifest)
            receipt_path=root/'annotations'/'receipts'/f"{ep['episode']:05d}_{digest(annotation)[:16]}.json"
            if not receipt_path.exists() or read_json(receipt_path).get('annotation_identity') != digest(annotation):
                raise ValueError('Verified annotation has no matching validated import receipt')
            usable[ep['episode']] = annotation
    return usable, receipt


def critical_mask(start, horizon, length, intervals, kind=None):
    """Union, not sum: overlapping phase labels cannot increase a step's weight."""
    t = np.arange(start, start + horizon)
    mask = np.zeros(horizon, dtype=bool)
    for item in intervals:
        if kind is not None and item['kind'] != kind:
            continue
        a, b = frame_interval(item['frames'], length)
        mask |= (t >= a) & (t < b)
    return mask & (t >= 0) & (t < length)


def cue_visible(start, interval):
    from vpp2.full_mask.curation import history_indices
    if interval['kind'] != 'memory_decision':
        return None
    if interval.get('cue_status') != 'identified':
        return None
    # Distinct cue intervals represent distinct REQUIRED observations. One anchor
    # frame cannot certify a cue that needs both an initial state and a transition.
    return all(any(a <= t < b for t in history_indices(start)) for a, b in interval['cue_intervals'])
