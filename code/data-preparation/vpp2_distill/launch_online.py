"""Supervise separate frozen-teacher and student processes inside one Slurm allocation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid

import yaml

from .contracts import write_json
from .checkpoints import source_identity


def gpu_groups(visible, teachers, students):
    devices = [d.strip() for d in visible.split(",") if d.strip()]
    if teachers < 1 or students < 1 or len(devices) != teachers+students or len(set(devices)) != len(devices):
        raise ValueError("Slurm-visible GPUs must exactly match the distinct teacher + student devices")
    if 32 % students:
        raise ValueError("Student process count must divide logical batch 32")
    return devices[:teachers], devices[teachers:]


def require_data_audit(config):
    root = Path(config["runtime"]["experiment"])
    report_path = root/"reports/cpu_preflight_complete.json"
    if not report_path.is_file():
        raise RuntimeError("Complete prepare_node and its full audit before launching GPU work")
    report = json.loads(report_path.read_text())
    data = report.get("checks", {}).get("data", {})
    manifest = json.loads((Path(config["data_root"])/"manifest.json").read_text())
    if not report.get("passed") or not data.get("full_byte_audit") or data.get("manifest") != manifest.get("sha256"):
        raise RuntimeError("Completed CPU/data audit does not match this prepared dataset")
    return report_path


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def student_command(python, config, accelerate_config, students, port, *, stage, output, urls,
                    resume=None, max_steps=None, stop_after=None, stop_file=None, refresh_round=1, rollouts=()):
    result = [python, "-m", "accelerate.commands.launch", "--config_file", str(accelerate_config),
              "--num_processes", str(students), "--main_process_port", str(port), "-m", "vpp2_distill.train",
              "--config", str(config), "--stage", stage, "--output", str(output)]
    for url in urls:
        result += ["--teacher-url", url]
    if resume:
        result += ["--resume", str(resume)]
    for flag, value in (("--max-steps", max_steps), ("--stop-after-updates", stop_after), ("--stop-file", stop_file)):
        if value is not None:
            result += [flag, str(value)]
    if stage == "refresh":
        result += ["--refresh-round", str(refresh_round)]
        for path in rollouts:
            result += ["--refresh-rollouts", str(path)]
    return result


def stop_owned_process(process):
    # Every child is created with start_new_session=True. Never signal a process
    # not created by this supervisor, or use a broad process-name kill.
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def wait_ready(process, url, instance, timeout):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Teacher exited during startup ({process.returncode}); inspect its log")
        try:
            with urllib.request.urlopen(url+"/metadata", timeout=2) as response:
                metadata = json.load(response)
            if metadata.get("instance_id") == instance:
                if not metadata["native_parity"]["passed"] or metadata["native_parity"]["checked_steps"] != 10:
                    raise RuntimeError("Native teacher parity did not pass")
                return metadata
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(2)
    raise TimeoutError("Teacher readiness timed out; inspect its startup/parity log")


def run(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use the prepared sbatch script; this launcher requires a Slurm allocation")
    repo = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    runtime = config["runtime"]
    teacher_gpus, student_gpus = gpu_groups(os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        int(runtime["teacher_processes"]), int(runtime["student_processes"]))
    experiment = Path(runtime["experiment"])
    require_data_audit(config)
    job = os.environ["SLURM_JOB_ID"]
    launch_id = f"{job}-{uuid.uuid4().hex[:8]}"
    logs = experiment/"logs"/launch_id
    logs.mkdir(parents=True, exist_ok=False)
    stop_file = logs/"stop_requested"
    stopping = False
    def request_stop(*_):
        nonlocal stopping
        stopping = True
        stop_file.touch()
        print("Stop requested; saving after the current complete logical update and validation", flush=True)
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", HF_HOME=str(experiment/"models"),
               PYTHONPATH=str(repo)+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else ""))
    if runtime.get("build_bin"):
        env["PATH"] = runtime["build_bin"]+os.pathsep+env.get("PATH", "")
    env["ACCELERATE_USE_CPU"] = "false"
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(name, None)
    # Run the CPU-only preflight in a separate process before loading any GPU model.
    report_path = logs/"preflight.json"
    subprocess.run([runtime["student_python"], "-m", "vpp2_distill.preflight", "--config", str(config_path),
                    "--report", str(report_path)], cwd=repo, env={**env,"CUDA_VISIBLE_DEVICES":""}, check=True)
    children, handles, urls, metadata, teachers = [], [], [], [], []
    launch = dict(job_id=job, host=socket.gethostname(), mode=args.mode, config=str(config_path),
                  teacher_gpus=teacher_gpus, student_gpus=student_gpus, logs=str(logs), commands=[], implementation=source_identity())
    write_json(logs/"launch.json", launch)
    try:
        for i, gpu in enumerate(teacher_gpus):
            port, instance = free_port(), uuid.uuid4().hex
            url = f"http://127.0.0.1:{port}"
            command = [runtime["teacher_python"], "-m", "vpp2_distill.teacher_service", "--vpp2-root", runtime["vpp2_root"],
                       "--data", config["data_root"], "--action", config["model"]["action_checkpoint"],
                       "--video", runtime["video_checkpoint"], "--wan", runtime["wan"], "--device", "cuda:0",
                       "--port", str(port), "--seed", str(config["seed"]), "--instance-id", instance,
                       "--metadata-output", str(logs/f"teacher_{i}_metadata.json"),
                       "--parity-fixture", str(logs/f"teacher_{i}_parity.pt"),
                       "--context-audit", str(logs/f"teacher_{i}_context_audit.json")]
            if runtime.get("require_native_contexts", False):
                command.append("--require-native-contexts")
            handle = (logs/f"teacher_{i}.log").open("w")
            handles.append(handle)
            process = subprocess.Popen(command, cwd=repo, env={**env,"CUDA_VISIBLE_DEVICES":gpu, "TORCH_EXTENSIONS_DIR":str(experiment/"build_cache/teacher")},
                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(process)
            teachers.append(process)
            launch["commands"].append(command)
            write_json(logs/"launch.json", launch)
            metadata.append(wait_ready(process, url, instance, float(runtime.get("startup_timeout", 1800))))
            urls.append(url)
            # Compare the native teacher against the actual student PyTorch runtime,
            # not merely against another wrapper in the teacher environment.
            parity_report = logs/f"student_runtime_parity_{i}.json"
            command = [runtime["student_python"], "-m", "vpp2_distill.action_parity",
                       "--fixture", str(logs/f"teacher_{i}_parity.pt"), "--action", config["model"]["action_checkpoint"],
                       "--report", str(parity_report), "--device", "cuda:0"]
            handle = (logs/f"student_runtime_parity_{i}.log").open("w")
            handles.append(handle)
            probe = subprocess.Popen(command, cwd=repo, env={**env,"CUDA_VISIBLE_DEVICES":student_gpus[0]},
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(probe)
            launch["commands"].append(command)
            write_json(logs/"launch.json", launch)
            if probe.wait(timeout=float(runtime.get("startup_timeout", 1800))) != 0:
                raise RuntimeError(f"Student-runtime native Action2B parity failed; inspect {parity_report}")
            if not json.loads(parity_report.read_text())["passed"]:
                raise RuntimeError("Student-runtime parity report did not pass")
            # The one-example diagnostic is disposable after passing; normal
            # training still requests every teacher target from the online model.
            (logs/f"teacher_{i}_parity.pt").unlink()
        if any(p.poll() is not None for p in teachers):
            raise RuntimeError("Teacher exited during the student-runtime parity check")
        if args.mode == "smoke":
            # Same full models and logical batch; only duration and validation panel shrink.
            smoke_config = {**config, "online_teacher":{**config["online_teacher"], "validation_anchors":2},
                            "checkpointing":{**config["checkpointing"], "keep_last":2}}
            config_path = logs/"smoke_config.yaml"
            config_path.write_text(yaml.safe_dump(smoke_config, sort_keys=False))
            output = experiment/"runs"/f"smoke-{launch_id}"
            stages = [dict(stage="warmup", output=output/"warmup", resume=None),
                      dict(stage="warmup", output=output/"warmup", resume=output/"warmup/step_0000001"),
                      dict(stage="main", output=output/"main", resume=output/"warmup/step_0000002"),
                      dict(stage="main", output=output/"main", resume=output/"main/step_0000001")]
        else:
            output = Path(args.output or experiment/"runs"/(args.stage if args.stage != "refresh" else f"refresh{args.refresh_round}"))
            stages = [dict(stage=args.stage, output=output, resume=args.resume)]
        for i, stage in enumerate(stages):
            if stopping:
                break
            command = student_command(runtime["student_python"], config_path,
                repo/"configs/vpp2_accelerate_online_zero2.yaml", len(student_gpus), free_port(),
                urls=urls, **stage, max_steps=2 if args.mode=="smoke" else None,
                stop_after=1 if args.mode=="smoke" else args.stop_after_updates,
                stop_file=stop_file, refresh_round=args.refresh_round, rollouts=args.refresh_rollouts)
            handle = (logs/f"student_{i}_{stage['stage']}.log").open("w")
            handles.append(handle)
            process = subprocess.Popen(command, cwd=repo, env={**env,"CUDA_VISIBLE_DEVICES":",".join(student_gpus), "TORCH_EXTENSIONS_DIR":str(experiment/"build_cache/student")},
                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(process)
            launch["commands"].append(command)
            write_json(logs/"launch.json", launch)
            while process.poll() is None:
                if any(p.poll() is not None for p in teachers):
                    raise RuntimeError("Frozen teacher worker exited during training; aborting this student job")
                time.sleep(2)
            if process.returncode:
                raise RuntimeError(f"Student stage exited with {process.returncode}; inspect {handle.name}")
        if args.mode == "smoke" and not stopping:
            completed = []
            for stage in ("warmup", "main"):
                path = output/stage/"step_0000002/progress.json"
                progress = json.loads(path.read_text())
                if progress["step"] != 2 or progress["total_steps"] != 2 or progress["stage"] != stage:
                    raise RuntimeError("Smoke checkpoint did not complete both updates")
                completed.append(str(path.parent))
            write_json(experiment/"reports"/f"gpu_smoke_{launch_id}.json",
                       dict(passed=True, checkpoints=completed, teacher_metadata=metadata, launch=str(logs/"launch.json"),
                            policy_performance_tested=False, scope="four-rank full-model warmup/main save and resume"))
        write_json(logs/"result.json", dict(passed=not stopping, stopped_after_update=stopping, output=str(output)))
        print(json.dumps({"logs":str(logs), "output":str(output), "stopped_after_update":stopping}), flush=True)
    finally:
        for process in reversed(children):
            stop_owned_process(process)
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/robodojo_qwen4b_action2b_online_node.yaml")
    parser.add_argument("--mode", choices=("smoke", "train"), required=True)
    parser.add_argument("--stage", choices=("warmup", "main", "refresh"), default="warmup")
    parser.add_argument("--output")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--refresh-round", type=int, choices=(1,2), default=1)
    parser.add_argument("--refresh-rollouts", action="append", default=[])
    run(parser.parse_args())


if __name__ == "__main__":
    main()
