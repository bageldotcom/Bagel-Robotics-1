#!/usr/bin/env python3
"""Install the preserved policy into a user-specified XPolicyLab checkout."""
import argparse
import hashlib
from pathlib import Path
import shutil
ROOT = Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xpolicylab',required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    checkout=Path(a.xpolicylab).expanduser().resolve()
    required = ('XPolicyLab.py', 'pyproject.toml', 'client_server/ws/model_server.py')
    missing = [name for name in required if not (checkout/name).is_file()]
    if missing:
        p.error('Expected an XPolicyLab checkout; missing: ' + ', '.join(missing))
    source=ROOT/'experiments/robodojo/policy/Bagel_Robotics_1'
    destination=checkout/'policy/Bagel_Robotics_1'
    if destination.exists():
        expected={str(x.relative_to(source)):hashlib.sha256(x.read_bytes()).hexdigest()
                  for x in source.rglob('*') if x.is_file() and '__pycache__' not in x.parts}
        if all((destination/n).is_file() and hashlib.sha256((destination/n).read_bytes()).hexdigest()==h
               for n,h in expected.items()):
            print('Matching adapter already installed: '+str(destination));return
        p.error('A different adapter is present. Use a separate checkout; existing files are preserved.')
    if a.dry_run:print(str(source)+' -> '+str(destination));return
    shutil.copytree(source,destination,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    print(destination)


if __name__=='__main__':main()
