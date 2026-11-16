#!/usr/bin/env python3
"""Launch an identity-checked archived stage; --dry-run never launches a job."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from release_tools import cli


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=cli.STAGES,required=True)
    p.add_argument('--workspace',required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--action',choices=['train','verify','resume-check','export','select','prune','verify-export'],default='train')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    return cli.main(['command' if a.dry_run else 'run','--workspace',a.workspace,
        '--stage',a.stage,'--image',a.image,'--action',a.action])


if __name__=='__main__':raise SystemExit(main())
