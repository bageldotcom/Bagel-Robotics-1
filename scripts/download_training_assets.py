#!/usr/bin/env python3
"""Download the pinned public base model/encoders and verify training hashes."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();m=json.loads((ROOT/'configs/artifacts.json').read_text())
    base=m['training_base'];out=Path(a.output).expanduser().resolve()
    if a.dry_run:
        print(json.dumps({'repo':base['repo'],'revision':base['revision'],'output':str(out),
            'files':list(base['files'])+[x['rfilename'] for x in m['inference']['wan']['files']]},indent=2));return
    if not a.verify_only:
        from huggingface_hub import hf_hub_download
        for name in list(base['files'])+[x['rfilename'] for x in m['inference']['wan']['files']]:
            hf_hub_download(base['repo'],name,revision=base['revision'],local_dir=out)
    for name,expected in base['files'].items():
        if sha(out/name)!=expected:raise ValueError('Training artifact checksum mismatch: '+name)
    from bagel_robotics.inference.assets import verify_wan
    verify_wan(out/'checkpoints/Wan2.1-I2V-14B-480P',full_hash=True)
    print('Verified public training base and encoder assets: '+str(out))


if __name__=='__main__':main()
