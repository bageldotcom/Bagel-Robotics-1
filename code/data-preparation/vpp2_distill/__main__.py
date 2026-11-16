"""Artifact utilities; no command launches training or simulator jobs implicitly."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--data", required=True)
    install = sub.add_parser("install-adapter")
    install.add_argument("--robodojo-root", required=True)
    audit = sub.add_parser("audit-data")
    audit.add_argument("--data", required=True)
    args = parser.parse_args()
    if args.command == "export":
        from .bundle import export_checkpoint
        export_checkpoint(args.checkpoint, args.output, args.data)
    elif args.command == "install-adapter":
        source = Path(__file__).resolve().parents[1] / "deployment/VPP2Student"
        target = Path(args.robodojo_root) / "XPolicyLab/policy/VPP2Student"
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        print(target)
    elif args.command == "audit-data":
        from .contracts import file_digest
        from .dataset import Demonstrations
        data = Demonstrations(args.data)
        for episode in data.manifest["episodes"]:
            if episode.get("text_sha256") and file_digest(data.root / episode["text"]) != episode["text_sha256"]:
                raise ValueError(f"Prepared text checksum mismatch: {episode['text']}")
            for name, expected in episode["files"].items():
                path = data.root / episode["path"] / name
                if file_digest(path) != expected:
                    raise ValueError(f"Prepared data checksum mismatch: {path}")
        print(json.dumps({"passed":True,"episodes":len(data.manifest["episodes"]),"manifest":data.manifest["sha256"]}))


if __name__ == "__main__":
    main()

