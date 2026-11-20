#!/usr/bin/env python3
"""Install the thin adapter and canonical source into a separate XPolicyLab checkout."""
import argparse
import hashlib
from pathlib import Path
import shutil
ROOT = Path(__file__).resolve().parents[1]
POLICY_NAME = "Bagel_Robotics_1"
IGNORED = {"__pycache__", ".pytest_cache", ".venv", ".bagel_source", "checkpoints"}


def source_files(directory):
    for path in directory.rglob("*"):
        relative = path.relative_to(directory)
        generated = any(part in IGNORED or part.startswith(".venv") or part.endswith(".egg-info")
                        for part in relative.parts)
        if path.is_file() and not generated and path.suffix != ".pyc":
            yield path, relative


def installation_files(root=ROOT):
    """Build one install payload from the canonical tree; no second tracked runtime."""
    root = Path(root)
    adapter = root / "benchmarks/robodojo/policy" / POLICY_NAME
    files = {relative: source for source, relative in source_files(adapter)}
    # Generate distribution copies from one canonical licensing set.
    files[Path("LICENSE")] = root / "LICENSE"
    files.update({Path("licenses") / relative: source
                  for source, relative in source_files(root / "licenses")})
    package = Path(".bagel_source")
    for folder in ("src", "configs", "licenses"):
        if (root / folder).is_dir():
            files.update({package / folder / relative: source
                          for source, relative in source_files(root / folder)})
    for name in ("pyproject.toml", "README.md", "LICENSE", "MANIFEST.in",
                 "requirements-inference.txt", "requirements-train.txt"):
        if (root / name).is_file():
            files[package / name] = root / name
    required = [Path("model.py"), Path("deploy.yml"), package / "pyproject.toml",
                package / "requirements-inference.txt",
                package / "src/bagel_robotics/inference/policy.py"]
    missing = [str(path) for path in required if path not in files]
    if missing:
        raise ValueError("Incomplete canonical source package: " + ", ".join(missing))
    return files


def same_file(left, right):
    return right.is_file() and hashlib.sha256(left.read_bytes()).digest() == hashlib.sha256(right.read_bytes()).digest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xpolicylab',required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    checkout=Path(a.xpolicylab).expanduser().resolve()
    required = ('XPolicyLab.py', 'pyproject.toml', 'client_server/ws/model_server.py')
    missing = [name for name in required if not (checkout/name).is_file()]
    if missing:
        p.error('Expected an XPolicyLab checkout; missing: ' + ', '.join(missing))
    files=installation_files()
    destination=checkout/'policy'/POLICY_NAME
    if destination.exists():
        if all(same_file(source,destination/relative) for relative,source in files.items()):
            print('Matching adapter and package already installed: '+str(destination));return
        p.error('A different adapter or package is present. Use a separate checkout; existing files are preserved.')
    if a.dry_run:print(f'{destination}: {len(files)} adapter/package files; no files written');return
    for relative,source in files.items():
        target=destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    print(destination)


if __name__=='__main__':main()
