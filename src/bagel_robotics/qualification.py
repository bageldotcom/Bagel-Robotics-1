"""Qualify one configured training stage with CPU contracts and a short GPU probe."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=['mask_learning', 'recovery'], required=True)
    args = parser.parse_args(argv)
    package = 'adaptive_pruning' if args.stage == 'mask_learning' else 'recovery'
    settings = importlib.import_module('bagel_robotics.' + package + '.config')
    config = settings.load(args.config)
    root = Path(config['root'])
    root.mkdir(parents=True, exist_ok=True)
    cpu = subprocess.run([sys.executable, '-m', 'bagel_robotics.selftest'],
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=False)
    if cpu.returncode:
        return cpu.returncode
    prepare = importlib.import_module('bagel_robotics.' + package + '.prepare')
    prepare.prepare(config)
    checks = root / 'checks'
    checks.mkdir(exist_ok=True)
    (checks / 'cpu.json').write_text(json.dumps({'passed': True, 'identity': settings.identity(config),
        'scope': 'Installed small-model CPU contracts; no full experiment'}, indent=2) + '\n')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=' + str(config['train']['world_size']),
               '-m', 'bagel_robotics.' + package + '.train', '--config', args.config,
               '--smoke-updates', '1']
    return subprocess.run(command, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
