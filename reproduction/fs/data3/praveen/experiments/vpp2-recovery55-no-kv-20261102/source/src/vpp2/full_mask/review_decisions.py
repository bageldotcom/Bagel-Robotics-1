"""Validate supplied curation annotations. Never invent reviewed phases or cues."""
import argparse
import json
from pathlib import Path
from .curation import load_panel
from .panel import immutable,file_hash


def validate_review(panel,review):
    if review.get('panel_identity')!=panel['identity']:raise ValueError('Review belongs to another panel')
    rows={r['window_id']:r for r in panel['windows']}
    supplied={}
    for d in review.get('decisions',[]):
        key=d['window_id']
        if key not in rows or key in supplied:raise ValueError('Unknown or duplicated review row')
        if d.get('episode')!=rows[key]['episode'] or d.get('frame')!=rows[key]['frame']:raise ValueError('Review location mismatch')
        if d.get('decision') not in ('unreviewed','retain_verified','retain_uncertain','replace_missing_phase','suggest_duplicate'):raise ValueError('Unknown review decision')
        supplied[key]=d
    protected=[r for r in rows.values() if r['protected_suite'] and r['role']=='selection_initial']
    pending=[]
    for r in protected:
        d=supplied.get(r['window_id'],{})
        if d.get('decision')!='retain_verified' or not d.get('evidence','').strip():pending.append(r['window_id'])
    changes=[d['window_id'] for d in supplied.values() if d['decision'] in ('replace_missing_phase','suggest_duplicate')]
    return dict(panel_identity=panel['identity'],passed=not pending and not changes,
                reviewed_protected=len(protected)-len(pending),required_protected=len(protected),
                pending_windows=pending,requested_replacements=changes,
                automatic_deletions=0,scope='Supplied review annotations; no simulator/test outcomes used.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--decisions',required=True);a=p.parse_args()
    c=json.loads(Path(a.config).read_text());panel=load_panel(c['panel_file']);review=json.loads(Path(a.decisions).read_text())
    receipt=validate_review(panel,review);receipt['review_file_sha256']=file_hash(a.decisions)
    name='visual_review_receipt.json' if receipt['passed'] else 'visual_review_pending.json'
    immutable(Path(c['curation']['output'])/name,receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('pending_windows','requested_replacements')},indent=2))


if __name__=='__main__':main()
