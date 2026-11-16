#!/usr/bin/env python3
"""CPU release audit: source identity, no-K/V recipe, adapter pins and safe files."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from release_tools import cli


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',help='Write a JSON check receipt')
    a=p.parse_args();errors=[]
    source=cli.verify_sources()
    if not source['passed']:errors.extend('source: '+x for x in source['failures'])
    for stage in cli.STAGES:
        try:cli.require_no_kv_training(stage,cli.read(cli.contained(ROOT/'reproduction/fs',cli.config_for(stage))))
        except (ValueError,OSError) as exc:errors.append(str(exc))
    policy=ROOT/'experiments/robodojo/policy/Bagel_Robotics_1'
    model=json.loads((policy/'artifact-manifest.json').read_text())
    artifacts=json.loads((ROOT/'configs/artifacts.json').read_text())
    if artifacts['inference']!=model:errors.append('Root and adapter checkpoint manifests differ')
    rules=[('HF token',re.compile(rb'hf_[A-Za-z0-9]{30,}')),
           ('GitHub token',re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{60,})')),
           ('private key',re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'))]
    checked=0
    for file in ROOT.rglob('*'):
        rel=file.relative_to(ROOT)
        if any(x in {'.git','outputs','__pycache__','.pytest_cache','checkpoints'} for x in rel.parts):continue
        if file.is_symlink():
            if not file.exists() or not file.resolve().is_relative_to(ROOT):errors.append('Invalid internal link: '+str(rel))
            continue
        if not file.is_file():continue
        checked+=1
        if file.stat().st_size>=100*1024**2:errors.append('Exceeds GitHub file limit: '+str(rel))
        if file.suffix in {'.py','.json','.md','.sh','.sbatch','.yaml','.yml','.txt'}:
            value=file.read_bytes()
            for label,rule in rules:
                if rule.search(value):errors.append(label+' pattern in '+str(rel))
    for file in (ROOT/'scripts').glob('*.sbatch'):
        result=subprocess.run(['bash','-n',str(file)],capture_output=True,text=True)
        if result.returncode:errors.append('Shell syntax: '+str(file.relative_to(ROOT)))
    receipt=dict(passed=not errors,errors=errors,source_files_checked=source['files_checked'],
        package_files_checked=checked,checkpoint_repo=model['repo_id'],checkpoint_revision=model['revision'],
        no_kv_training_enforced=True,gpu_training_rerun=False,public_release_complete=False)
    if a.output:Path(a.output).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2));return 0 if receipt['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
