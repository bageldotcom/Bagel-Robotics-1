"""Re-score saved native action predictions as verified annotations arrive."""
import argparse
import json
from pathlib import Path
import numpy as np
from vpp2.full_mask.panel import digest, immutable, file_hash
from .contracts import read_json, checked_identity, load_annotations
from .metrics import native_queries, action_errors, score_chunk, aggregate


def score(config, prediction_dir, allow_partial=False):
    root=Path(config['root']);manifest=checked_identity(read_json(root/'audit_manifest.json'))
    annotations,annotation_receipts=load_annotations(config,manifest)
    directory=Path(prediction_dir);ident=read_json(directory/'identity.json')
    if ident['manifest_identity']!=manifest['identity']:raise ValueError('Prediction panel differs')
    rows=[];missing=[];expected=0
    for e in manifest['episodes']:
        if e['role']!=ident['role']:continue
        for t in native_queries(e['length']):
            expected+=1;path=directory/f"{e['episode']:05d}_{t:05d}.npz"
            if not path.exists():missing.append([e['episode'],t]);continue
            with np.load(path,allow_pickle=False) as z:
                if str(z['identity'])!=ident['identity'] or int(z['episode'])!=e['episode'] or int(z['frame'])!=t:
                    raise ValueError('Prediction identity mismatch')
                if str(z['task'])!=e['task'] or str(z['suite'])!=e['suite'] or int(z['valid_steps'])!=min(32,e['length']-t):
                    raise ValueError('Prediction metadata mismatch')
                errors=action_errors(z['normalized'],z['target_normalized'],z['physical'],z['target_physical'])
            rows.append(dict(variant=ident['variant'],task=e['task'],suite=e['suite'],episode=e['episode'],frame=t,
                             scores=score_chunk(t,e['length'],errors,annotations.get(e['episode']))))
    if missing and not allow_partial:raise ValueError(f'{len(missing)} of {expected} inference queries missing; do not report a complete score')
    report=aggregate(rows)
    report.update(prediction_identity=ident['identity'],manifest_identity=manifest['identity'],
                  metric_source_identity=digest({name:file_hash(Path(__file__).with_name(name)) for name in ('score.py','metrics.py','contracts.py')}),
                  expected_queries=expected,completed_queries=len(rows),missing_queries=missing,
                  predictions_complete=not missing,annotation_receipts=annotation_receipts,
                  verified_evaluated_episodes=sum(e['role']==ident['role'] and e['episode'] in annotations for e in manifest['episodes']),
                  policy_retention_established=False)
    report['identity']=digest(report)
    out=root/'reports'/f"action_metrics_{report['identity'][:12]}.json";immutable(out,report)
    return out,report


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--predictions');g.add_argument('--latest-native',action='store_true')
    p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    config=read_json(a.config)
    if a.latest_native:
        from .inference import source_identity
        source=source_identity();matches=[]
        for pth in (Path(config['root'])/'predictions').glob('*/identity.json'):
            item=read_json(pth)
            if item.get('source_identity')==source and item.get('variant')=='native' and item.get('role')=='monitor':
                matches.append(pth.parent)
        if len(matches)!=1:raise ValueError('Expected exactly one native prediction run matching the current source')
        a.predictions=matches[0]
    path,result=score(config,a.predictions,a.allow_partial)
    print(json.dumps(dict(report=str(path),complete=result['predictions_complete'],queries=result['completed_queries'],
                         verified_episodes=result['verified_evaluated_episodes'])))


if __name__=='__main__':main()
