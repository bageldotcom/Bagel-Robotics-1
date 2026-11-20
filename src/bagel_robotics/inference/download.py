"""Download and verify a configured policy bundle and its encoder assets."""
from pathlib import Path
import argparse
import json
import shutil

from .assets import MANIFEST, file_digest, load_manifest, verify_bundle, verify_wan


def main(argv=None, *, policy_dir=None):
    policy_dir = Path(policy_dir or Path.cwd())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Explicit artifact manifest; defaults to the released model")
    parser.add_argument("--bundle", type=Path, default=policy_dir / "checkpoints/default")
    parser.add_argument("--wan-root", type=Path, default=policy_dir / "checkpoints/Wan2.1-I2V-14B-480P")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Download/check small checkpoint metadata without tensors or encoders")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest) if args.manifest else MANIFEST
    if args.metadata_only and args.verify_only:
        parser.error("Choose metadata-only or verify-only")
    if not args.verify_only:
        from huggingface_hub import hf_hub_download
        for filename in manifest["files"]:
            if args.metadata_only and filename in {"action.pt", "video.pt"}:
                continue
            hf_hub_download(manifest["repo_id"], filename, revision=manifest["revision"],
                            local_dir=args.bundle)
        if args.metadata_only:
            for name, expected in manifest["files"].items():
                if name not in {"action.pt", "video.pt"}:
                    path = args.bundle / name
                    if path.stat().st_size != expected["size"] or file_digest(path) != expected["sha256"]:
                        raise ValueError(f"Metadata mismatch: {name}")
            print("Pinned metadata verified. Tensor weights and encoders are not checked.")
            return
        prefix = "checkpoints/Wan2.1-I2V-14B-480P/"
        for entry in manifest["wan"]["files"]:
            source = hf_hub_download(manifest["wan"]["repo"], entry["rfilename"],
                                     revision=manifest["wan"]["revision"])
            destination = args.wan_root / entry["rfilename"].removeprefix(prefix)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(source, destination)
    print(json.dumps({"checkpoint": verify_bundle(args.bundle, manifest=manifest),
                      "encoders": verify_wan(args.wan_root, manifest=manifest)}, indent=2))


if __name__ == "__main__":
    main()
