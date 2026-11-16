#!/usr/bin/env python3
"""Push this prepared source only to the specified private Bagel repository."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
TARGET='bageldotcom/Bagel-Robotics-1'
URL='https://github.com/'+TARGET+'.git'


def run(*args):
    return subprocess.check_output(list(args),cwd=ROOT,text=True).strip()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--push',action='store_true',help='Push the prepared local main branch after checks')
    a=p.parse_args()
    info=json.loads(run('gh','repo','view',TARGET,'--json','nameWithOwner,isPrivate,viewerPermission'))
    if info['nameWithOwner'].lower()!=TARGET.lower() or not info['isPrivate']:
        raise SystemExit('Target must be the authorized private repository')
    if info['viewerPermission'] not in ['WRITE','MAINTAIN','ADMIN']:
        raise SystemExit('GitHub Write access is required')
    subprocess.run([sys.executable,'scripts/verify_release.py'],cwd=ROOT,check=True)
    if run('git','status','--porcelain'):
        raise SystemExit('Commit and review the local changes before pushing')
    if run('git','branch','--show-current')!='main':raise SystemExit('Expected prepared main branch')
    if not a.push:
        print('Private target and write access verified. Use --push to upload the reviewed commit.');return
    # No force push, repository creation, visibility change, PR creation or release publication.
    subprocess.run(['git','push',URL,'HEAD:refs/heads/main'],cwd=ROOT,check=True)
    print('Pushed to private '+TARGET)


if __name__=='__main__':main()
