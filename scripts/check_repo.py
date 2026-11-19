"""Check maintained source, interfaces, links and metadata without model execution."""
import ast
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    errors = []
    files = [p for p in ROOT.rglob('*') if p.is_file() and not any(
        part in {'.git', 'local', '__pycache__', 'build', 'dist', '.pytest_cache'} or part.startswith('.venv') or part.endswith('.egg-info')
        for part in p.relative_to(ROOT).parts)]
    for path in files:
        if path.suffix == '.py':
            try:
                ast.parse(path.read_text(), filename=str(path))
            except SyntaxError as exc:
                errors.append(str(exc))
        elif path.suffix in {'.sh', '.sbatch'}:
            check = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
            if check.returncode:
                errors.append(check.stderr)
        if path.is_symlink() and (not path.exists() or not path.resolve().is_relative_to(ROOT)):
            errors.append('Invalid symlink: ' + str(path))
    for path in [ROOT / 'README.md'] + list((ROOT / 'docs').glob('*.md')):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if target.startswith(('http:', 'https:', '#')):
                continue
            if not (path.parent / target.split('#')[0]).exists():
                errors.append('Missing documentation target: ' + str(path.relative_to(ROOT)) + ': ' + target)
    if json.loads((ROOT / 'configs/artifacts.json').read_text())['inference'] != json.loads(
            (ROOT / 'src/bagel_robotics/inference/configs/artifacts.json').read_text()):
        errors.append('Inference model manifest differs from public configuration')
    for recipe in (ROOT / 'configs/recipes').glob('*.json'):
        packaged = ROOT / 'src/bagel_robotics/resources/recipes' / recipe.name
        if not packaged.is_file() or recipe.read_bytes() != packaged.read_bytes():
            errors.append('Packaged recipe differs from public configuration: ' + recipe.name)
    result = {'passed': not errors, 'errors': errors, 'files_checked': len(files),
              'scope': 'Source and repository integrity; no training or benchmark rerun'}
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
