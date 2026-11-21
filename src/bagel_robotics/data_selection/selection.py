"""Materialize the published demonstration-window selection without resampling."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
from ..configuration import resource, read_json

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def validate_panel(panel):
    rows=panel['windows']
    fit=[r for r in rows if r['role']=='selection_initial']
    monitor=[r for r in rows if r['role']=='monitor']
    if len(fit)!=30000 or len(monitor)!=1119 or len(panel['tasks'])!=35:
        raise ValueError('Pinned panel counts changed')
    episodes={role:{p['episode'] for p in panel['episode_partitions'] if p['role']==role}
              for role in ('fit','monitor','audit')}
    if [len(episodes[k]) for k in episodes] != [2906,280,280]:
        raise ValueError('Episode partition changed')
    if len(set.union(*episodes.values())) != sum(map(len,episodes.values())):
        raise ValueError('Episode leakage')
    if set.union(*episodes.values()) & set(panel['official_validation_episodes']):
        raise ValueError('Official validation leakage')
    if len({r['id'] for r in rows})!=len(rows) or len({(r['episode'],r['frame']) for r in rows})!=len(rows):
        raise ValueError('Duplicate windows')
    for subset,role in ((fit,'fit'),(monitor,'monitor')):
        if {r['episode'] for r in subset} != episodes[role]:
            raise ValueError('Missing/forbidden episode in '+role)
        if {r['task'] for r in subset} != set(panel['tasks']):
            raise ValueError('Missing task')
        if any(not 0<=r['frame']<=r['length']-32 for r in subset):
            raise ValueError('Incomplete action chunk')
    return fit,monitor

def load_panel():
    raw = gzip.decompress(resource("selected_panel.json.gz").read_bytes())
    expected = read_json(resource("data.json"))["panel_sha256"]
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Published selection checksum changed")
    panel = json.loads(raw)
    if digest({k:v for k,v in panel.items() if k!="identity"}) != panel["identity"]:
        raise ValueError("Published selection identity changed")
    return panel

def materialize(output, dry_run=False, data_root=None):
    panel = load_panel()
    fit, monitor = validate_panel(panel)
    if data_root:
        expected = gzip.decompress(resource("native_manifest.json.gz").read_bytes())
        if (Path(data_root)/"manifest.json").read_bytes() != expected:
            raise ValueError("Native dataset manifest differs from the published source")
    summary = dict(panel_identity=panel["identity"], fitting_windows=len(fit),
        monitoring_windows=len(monitor), fitting_episodes=len({r["episode"] for r in fit}),
        monitoring_episodes=len({r["episode"] for r in monitor}), tasks=len(panel["tasks"]),
        split_leakage=False, selection="Published panel; no resampling")
    if dry_run:
        return summary
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty selection output directory")
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in (("fitting",fit),("monitoring",monitor)):
        path = output/(name+".jsonl")
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with (output/(name+".csv")).open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=["id","episode","frame","length","task","role"])
            writer.writeheader()
            writer.writerows({key:row[key] for key in writer.fieldnames} for row in rows)
    (output/"selection.json").write_text(json.dumps(dict(summary,files=files),indent=2)+"\n")
    return summary
