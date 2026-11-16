"""Resumable CPU-only node preparation followed by a full data/model audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess

import yaml

from .contracts import write_json


def run(args):
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    runtime = config["runtime"]
    experiment = Path(runtime["experiment"])
    reports, logs = experiment/"reports", experiment/"logs"
    reports.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES":"", "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1",
           "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1", "PYTHONUNBUFFERED":"1"}
    env["PYTHONPATH"] = str(repo)+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    commands = [
        ("bootstrap", [runtime["teacher_python"], "-m", "vpp2_distill.bootstrap_data",
            "--vpp2-root", runtime["vpp2_root"], "--source", runtime["source"], "--wan", runtime["wan"],
            "--stats", str(Path(config["model"]["action_checkpoint"]).parent/"dataset_stats.json"),
            "--output", runtime["prepared_split"], "--device", "cpu", "--threads", "2"]),
        ("frames", [runtime["teacher_python"], "-m", "vpp2_distill.prepare",
            "--vpp2-root", runtime["vpp2_root"], "--source", runtime["source"],
            "--prepared", runtime["prepared_split"], "--output", config["data_root"],
            "--workers", str(args.workers), "--threads", "1", "--min-free-gib", "256"]),
        ("audit", [runtime["student_python"], "-m", "vpp2_distill.preflight", "--config", str(config_path),
            "--full-data", "--load-models", "--report", str(reports/"cpu_preflight_complete.json")])]
    with (experiment/"preparation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(running=True, passed=False, pid=os.getpid(), workers=args.workers,
                      started=datetime.now(timezone.utc).isoformat(), completed_phases=[], data=config["data_root"])
        try:
            for phase, command in commands:
                status.update(phase=phase, command=command)
                write_json(reports/"preparation_status.json", status)
                print(json.dumps({"phase":phase,"command":command}), flush=True)
                with (logs/f"prepare_{phase}.log").open("a") as handle:
                    handle.write(json.dumps({"started":datetime.now(timezone.utc).isoformat(),"pid":os.getpid()})+"\n")
                    handle.flush()
                    subprocess.run(command, cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
                status["completed_phases"].append(phase)
            status.update(running=False,passed=True,phase="complete",finished=datetime.now(timezone.utc).isoformat())
        except BaseException as exc:
            status.update(running=False,passed=False,error=str(exc),finished=datetime.now(timezone.utc).isoformat())
            raise
        finally:
            write_json(reports/"preparation_status.json", status)
        print(json.dumps(status), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/robodojo_qwen4b_action2b_online_node.yaml")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    run(args)


if __name__ == "__main__":
    main()
