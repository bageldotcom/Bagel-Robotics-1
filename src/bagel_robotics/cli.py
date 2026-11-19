"""Small, lazy command interface to the released Bagel Robotics workflow."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .configuration import load, resolve, resource, read_json, selected_recipe


def _data_args(config, args):
    data = config.get("data", {})
    output = resolve(config, data.get("output"))
    if output is None:
        raise ValueError("Set data.output in your configuration")
    result = ["--output", str(output)]
    for field, flag in (("source", "--source"), ("reader", "--vpp2-root")):
        value = resolve(config, data.get(field))
        if value:
            result += [flag, str(value)]
    for field, flag in (("workers", "--workers"), ("threads", "--threads"), ("min_free_gib", "--min-free-gib")):
        if field in data:
            result += [flag, str(data[field])]
    if args.episodes:
        result += ["--episodes"] + [str(x) for x in args.episodes]
    if args.dry_run:
        result += ["--dry-run"]
    if args.verify_only:
        result += ["--verify-only"]
    return result


def evaluation_commands(config, operation):
    paths = config.get("paths", {})
    policy = resolve(config, paths.get("policy"))
    if policy is None:
        raise ValueError("Set paths.policy to the installed RoboDojo adapter directory")
    python = resolve(config, paths.get("python"), follow_symlinks=False)
    if python is None:
        raise ValueError("Set paths.python to the policy environment's Python executable")
    if operation == "download":
        return [{"argv": [str(python), "-m", "bagel_robotics.inference.download", "--bundle",
                    str(resolve(config, paths.get("bundle")) or policy / "checkpoints/default"),
                    "--wan-root", str(resolve(config, paths.get("encoders")) or policy / "checkpoints/Wan2.1-I2V-14B-480P")],
                 "env": {}, "cwd": str(policy)}]
    settings = config.get("evaluation", {})
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "ALLOCATED_GPU")
    if "," in gpu:
        raise ValueError("The inference smoke command expects one allocated GPU")
    commands = []
    for encoded in ("0", "1"):
        env = {"EVAL_ENV_TYPE": "debug", "DEBUG_OBS_ENCODED": encoded, "DEBUG_EVAL_EPISODES": "1"}
        for key, variable in (("bundle", "BAGEL_BUNDLE"), ("encoders", "BAGEL_WAN_ROOT")):
            value = resolve(config, paths.get(key))
            if value:
                env[variable] = str(value)
        argv = ["bash", str(policy / "eval.sh"), "RoboDojo", settings.get("task", "stack_blocks"),
                "default", "arx_x5", "ee", str(settings.get("seed", 1)), gpu, gpu,
                str(python.parent.parent), str(python.parent.parent)]
        commands.append({"argv": argv, "env": env, "cwd": str(policy)})
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in ("init", "doctor", "prepare-data", "select-data", "learn-masks", "prune", "recover", "evaluate"):
        item = sub.add_parser(stage)
        item.add_argument("--config", required=True, help="Portable research/evaluation JSON configuration")
        if stage not in {"init", "doctor"}:
            item.add_argument("--dry-run", action="store_true", help="Inspect without creating outputs or launching a process")
        if stage == "doctor":
            item.add_argument("--stage", dest="check_stage", choices=["learn-masks", "recover"], default="learn-masks")
            item.add_argument("--full-hash", action="store_true")
        if stage == "prepare-data":
            item.add_argument("--episodes", nargs="+", type=int)
            item.add_argument("--verify-only", action="store_true")
        if stage in {"learn-masks", "prune", "recover"}:
            actions = {"learn-masks": ["prepare", "verify", "train"], "prune": ["select", "export", "verify"],
                       "recover": ["verify", "train"]}[stage]
            item.add_argument("--action", choices=actions, default="export" if stage == "prune" else "train")
        if stage == "evaluate":
            item.add_argument("--operation", choices=["smoke", "download"], default="smoke")
    args = parser.parse_args(argv)
    try:
        config = load(args.config)
        if args.stage == "init":
            from .workspace import init
            result = init(config)
        elif args.stage == "doctor":
            from .workspace import doctor
            result = doctor(config, args.check_stage, args.full_hash)
            print(json.dumps(result, indent=2))
            return 0 if result["passed"] else 1
        elif args.stage == "prepare-data":
            from .data_selection.prepare_dataset import main as prepare
            return prepare(_data_args(config, args)) or 0
        elif args.stage == "select-data":
            from .data_selection.selection import materialize
            data = config.get("data", {})
            output = resolve(config, data.get("selection_output"))
            if output is None:
                raise ValueError("Set data.selection_output in your configuration")
            result = materialize(output, args.dry_run)
        elif args.stage in {"learn-masks", "prune", "recover"}:
            from .workspace import command, run
            if args.dry_run:
                try:
                    cmd = command(config, args.stage, args.action)
                    result = {"stage": args.stage, "action": args.action, "argv": cmd, "command": shlex.join(cmd), "launched": False}
                except ValueError as exc:
                    result = {"stage": args.stage, "action": args.action, "recipe": selected_recipe(config, args.stage),
                              "configuration_needed": str(exc), "launched": False}
            else:
                return run(config, args.stage, args.action)
        else:
            commands = evaluation_commands(config, args.operation)
            if args.dry_run:
                result = {"operation": args.operation, "commands": commands, "launched": False}
            else:
                if args.operation == "smoke" and not os.environ.get("SLURM_JOB_ID"):
                    raise ValueError("Run the GPU smoke check inside a Slurm allocation")
                for entry in commands:
                    if not Path(entry["cwd"]).is_dir():
                        raise ValueError("Install the benchmark adapter first: " + entry["cwd"])
                    process = subprocess.run(entry["argv"], cwd=entry["cwd"], env=dict(os.environ, **entry["env"]), check=False)
                    if process.returncode:
                        return process.returncode
                return 0
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
