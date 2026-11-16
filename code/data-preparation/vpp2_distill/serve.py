"""Launch the student through the pinned VPP2 XPolicyLab server interface."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def command(args):
    root = Path(args.robodojo_root).resolve()
    bundle = Path(args.bundle).resolve()
    entry = root / "XPolicyLab/setup_policy_server.py"
    config = root / "XPolicyLab/policy/VPP2Student/deploy.yml"
    for path in (entry, config, bundle/"manifest.json", bundle/"dataset_stats.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if json.loads((bundle/"manifest.json").read_text())["format"] != "vpp2-student-bundle-v1":
        raise ValueError("Expected an exported student bundle")
    if bool(args.rollout_output) != bool(args.development_tag):
        raise ValueError("Development rollout output and development_tag must be supplied together")
    overrides = {"host":args.host, "port":args.port, "checkpoint_path":str(bundle),
                 "dataset_stats_path":str(bundle/"dataset_stats.json"), "ckpt_name":args.ckpt_name,
                 "action_horizon":32, "replan_steps":24, "num_inference_steps":10, "sigma_shift":1., "seed":1}
    if args.text_service_url:
        overrides["text_service_url"] = args.text_service_url
    if args.rollout_output:
        overrides.update(rollout_output=str(Path(args.rollout_output).resolve()),development_tag=args.development_tag)
    return [sys.executable,"-u",str(entry),"--config_path",str(config),"--overrides"] + [f"{k}={v}" for k,v in overrides.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo-root", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--port", type=int, default=19961)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--ckpt-name", default="vpp2_student")
    parser.add_argument("--text-service-url")
    parser.add_argument("--rollout-output")
    parser.add_argument("--development-tag")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cmd = command(args)
    print(json.dumps({"command":cmd,"gpu":args.gpu},indent=2),flush=True)
    if args.dry_run:
        return
    root = Path(args.robodojo_root).resolve()
    repository = Path(__file__).resolve().parents[1]
    env = {**os.environ,"CUDA_VISIBLE_DEVICES":args.gpu,
           "PYTHONPATH":os.pathsep.join((str(repository),str(root),str(root/"XPolicyLab"),os.environ.get("PYTHONPATH","")))}
    subprocess.run(cmd,cwd=root,env=env,check=True)


if __name__ == "__main__":
    main()
