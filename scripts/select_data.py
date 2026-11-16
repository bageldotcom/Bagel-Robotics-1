#!/usr/bin/env python3
"""Materialize the exact fitting/monitoring selections, rejecting split leakage."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--data-root',help='Optionally verify the prepared native dataset manifest')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    panel_path=ROOT/'data/selected-panel.json'
    raw=panel_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='bc6df13b64b5f96bbe17c64244240cc44a40d704babb70893f79c0f30d389680':
        raise ValueError('Selected panel checksum changed')
    panel=json.loads(raw)
    if digest({k:v for k,v in panel.items() if k!='identity'})!=panel['identity']:
        raise ValueError('Panel identity is inconsistent')
    sys.path.insert(0,str(ROOT/'code/mask-no-kv/src'))
    from vpp2.flexible_mask.data import validate_panel
    fit,monitor=validate_panel(panel)
    if a.data_root:
        expected=(ROOT/'data/native-manifest.json').read_bytes()
        if (Path(a.data_root)/'manifest.json').read_bytes()!=expected:
            raise ValueError('Native dataset manifest differs from the training source')
    summary=dict(panel_identity=panel['identity'],fitting_windows=len(fit),monitoring_windows=len(monitor),
        fitting_episodes=len({r['episode'] for r in fit}),monitoring_episodes=len({r['episode'] for r in monitor}),
        tasks=len(panel['tasks']),split_leakage=False,selection='Exact published panel; no resampling')
    if a.dry_run:print(json.dumps(summary,indent=2));return
    out=Path(a.output).expanduser().resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    files={}
    for name,rows in [('fitting',fit),('monitoring',monitor)]:
        f=out/(name+'.jsonl')
        with f.open('w') as h:
            for row in rows:h.write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n')
        files[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
        with (out/(name+'.csv')).open('w',newline='') as h:
            w=csv.DictWriter(h,fieldnames=['id','episode','frame','length','task','role']);w.writeheader()
            w.writerows({k:row[k] for k in w.fieldnames} for row in rows)
    (out/'selection.json').write_text(json.dumps(dict(summary,files=files),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
