"""Independent split, artifact and sample-coverage checks for v3."""
from collections import Counter
import json
from pathlib import Path
from .panel import digest,file_hash


def verify_roles(config,panel):
    out=Path(config.get('curation_artifacts',config['curation']['output']));inv=json.loads((out/'inventory.json').read_text())
    if digest({k:v for k,v in inv.items() if k!='identity'})!=inv['identity']:raise ValueError('Inventory modified')
    if panel['inventory_identity']!=inv['identity']:raise ValueError('Inventory mismatch')
    parts={e['episode']:e for e in inv['episodes']}
    if Counter(e['role'] for e in parts.values())!={'fit':2906,'monitor':280,'audit':280}:raise ValueError('Split counts changed')
    for r in panel['windows']:
        expected='fit' if r['role']=='selection_initial' else 'monitor' if r['role']=='monitor' else None
        if expected is None or parts[r['episode']]['role']!=expected:raise ValueError('Episode role leak')
    selected={r['episode'] for r in panel['windows'] if r['role']=='selection_initial'}
    expected={e for e,x in parts.items() if x['role']=='fit'}
    if selected!=expected:raise ValueError('Broad episode coverage incomplete')
    for e,x in parts.items():
        if x['role']=='audit':
            for directory in ('candidates','visual','summaries'):
                if (out/directory/f'{e:05d}.npz').exists():raise ValueError('Audit content entered feature/quality stages')
    for key,name in [('quality_receipt_sha256','quality_receipt.json'),('visual_receipt_sha256','visual_features_receipt.json')]:
        if panel[key]!=file_hash(out/name):raise ValueError('Feature/quality artifact changed')
    quality=json.loads((out/'quality_receipt.json').read_text())
    if {int(k) for k in quality['episodes']}!=expected:raise ValueError('Quality scoring episode scope changed')
    if quality['audit_episodes_used']!=0:raise ValueError('Audit used in quality fitting')
    expected_features={e for e,x in parts.items() if x['role']!='audit'}
    if {int(k) for k in quality['input_cache_hashes']}!=expected_features:raise ValueError('Feature/quality inputs changed')
    for ep,hashes in quality['input_cache_hashes'].items():
        for name,expected_hash in hashes.items():
            if file_hash(out/name/f'{int(ep):05d}.npz')!=expected_hash:raise ValueError('Feature or candidate bytes changed after quality fitting')
    if config['train']['frozen_video'] is not True or config['train']['frozen_action'] is not True:
        raise ValueError('Approved frozen mask-learning behavior changed')
    if config['approved_video_parameters']!=8121958978:raise ValueError('Approved parameter budget changed')
    return dict(passed=True,fit_episodes=len(selected),monitor_episodes=280,audit_episodes=280,
                audit_sealed=True,quality_training_only=True)
