"""Portable orchestration; never patches training mathematics or source snapshots."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNS = json.loads((ROOT / "reproduction/runs.json").read_text())
STAGES = {
    "mask-no-kv": ("mask_no_kv", "vpp2.flexible_mask.train"),
    "export-1500": ("compact_parent", "control/export_selected.py"),
    "recovery-55-to-250": ("recovery_pilot", "vpp2.flexible_mask.low_lr_entry"),
    "recovery-250-to-1000": ("recovery_1000", "vpp2.flexible_mask.recovery_extension_entry"),
}
PREFIXES = ["/data1/praveen/experiments", "/data3/praveen/experiments"]
COMMON = ["demonstrations", "wan", "teacher"]
NEEDS = {
    "mask-no-kv": COMMON + ["initial_rankings_no_kv"],
    "export-1500": COMMON + ["selected1500"],
    "recovery-55-to-250": COMMON + ["compact1500", "checkpoint55"],
    "recovery-250-to-1000": COMMON + ["compact1500", "checkpoint55", "checkpoint250"],
}

def read(path):
    return json.loads(Path(path).read_text())

def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def contained(root, logical):
    p = PurePosixPath(logical)
    if not p.is_absolute() or ".." in p.parts:
        raise ValueError("Expected a safe absolute container path: " + str(logical))
    root = Path(root).resolve()
    target = root.joinpath(*p.parts[1:])
    if not target.resolve().is_relative_to(root):
        raise ValueError("Path escapes workspace: " + str(logical))
    return target

def require_no_kv_training(stage, config):
    """The release permits no auxiliary K/V-matching loss in training."""
    if stage == "export-1500":
        return  # Tensor export does not optimize a training loss.
    value = config.get("train", {}).get("kv_weight")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != 0:
        raise ValueError("Released training requires train.kv_weight=0: " + stage)


def config_for(stage):
    key, _ = STAGES[stage]
    return RUNS[key]["root"] + "/config.json"

def workspace(path):
    p = Path(path).expanduser().resolve()
    if not (p / "workspace.json").is_file():
        raise ValueError("Run init first: " + str(p))
    meta = read(p / "workspace.json")
    if meta.get("schema") != "vpp2-mask-release-workspace-v1":
        raise ValueError("Unsupported workspace")
    return p

def verify_sources():
    failures = []
    manifest = read(ROOT / "reproduction/release_files.json")
    additions = ROOT / "reproduction/additional_files.json"
    if additions.exists(): manifest.update(read(additions))
    for name, rec in manifest.items():
        p = contained(ROOT, "/" + name)
        if not p.is_file() or p.stat().st_size != rec["bytes"] or file_hash(p) != rec["sha256"]:
            failures.append(name)
    return dict(files_checked=len(manifest), passed=not failures, failures=failures)

def init_workspace(path, local=False):
    p = Path(path).expanduser().resolve()
    if p == ROOT or p.is_relative_to(ROOT / "reproduction"):
        raise ValueError("Use a new workspace outside the archived sources")
    if p.exists() and any(p.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty directory: " + str(p))
    p.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "reproduction/fs", p / "fs", symlinks=False)
    # Archived receipts are evidence, not already-completed outputs of a new replay.
    for run_key in RUNS:
        for stage_dir in ("mask", "recovery"):
            old = contained(p / "fs", RUNS[run_key]["root"] + "/" + stage_dir + "/checkpoints")
            if old.exists(): shutil.rmtree(old)
    assets = read(ROOT / "configs/assets.example.json")
    for rec in assets.values():
        rec["host_path"] = rec.pop("original_path") if local else None
    if not local:
        assets["initial_rankings_no_kv"]["host_path"] = str(ROOT / "assets/initial-rankings-no-kv.pt")
    write(p / "assets.json", assets)
    write(p / "workspace.json", dict(schema="vpp2-mask-release-workspace-v1",
        package=str(ROOT), source_manifest_sha256=file_hash(ROOT / "reproduction/release_files.json"),
        weights_included=False, uploads=False))
    # The archived pilot uses symlinks to a separate checkpoint volume.
    roots = [RUNS[k]["root"] for k in ("recovery_pilot", "recovery_1000")]
    for logical_root in roots:
        cfg = read(contained(p / "fs", logical_root + "/config.json"))
        storage = cfg["recovery_lr_branch"]["artifact_storage"]
        for name in ("checkpoints", "best_models", "exports"):
            target = contained(p / "fs", storage + "/" + name)
            target.mkdir(parents=True, exist_ok=True)
            link = contained(p / "fs", logical_root + "/recovery/" + name)
            if link.exists():
                if any(link.iterdir()):
                    # Metadata-only parent receipts are moved into the storage directory.
                    for item in list(link.iterdir()):
                        shutil.move(str(item), str(target / item.name))
                link.rmdir()
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(os.path.relpath(storage + "/" + name, logical_root + "/recovery"), target_is_directory=True)
        contained(p / "fs", cfg["recovery_lr_branch"]["load_scratch"]).mkdir(parents=True, exist_ok=True)
    (p / "logs").mkdir()
    print(str(p))
    return p

def assets_for(p, stage):
    raw = read(p / "assets.json")
    result = {}
    for name in NEEDS[stage]:
        rec = dict(raw[name])
        host = rec.get("host_path")
        # After a replay of the pilot, use its newly generated checkpoint250.
        if name in {"checkpoint250", "mask31", "mask47", "mask1251"} and not host:
            generated = contained(p / "fs", rec["container_path"])
            if (generated / "COMPLETE.json").is_file():
                rec["host_path"] = str(generated)
        result[name] = rec
    return result

def check_asset(name, rec, full_hash=False):
    host = rec.get("host_path")
    if not host:
        return [name + ": set host_path in assets.json"]
    p = Path(host).expanduser()
    if not p.is_absolute() or not p.exists():
        return [name + ": missing absolute host path: " + str(p)]
    errors = []
    for rel, expected in rec.get("metadata_sha256", {}).items():
        f=contained(p,"/"+rel)
        if not f.is_file() or file_hash(f)!=expected: errors.append(name+": metadata hash mismatch "+rel)
    if full_hash:
        for rel, expected in rec.get("weight_sha256", {}).items():
            f=contained(p,"/"+rel)
            if not f.is_file() or file_hash(f)!=expected: errors.append(name+": model hash mismatch "+rel)
    for rel in rec.get("required_files", []):
        if not (p / rel).is_file():
            errors.append(name + ": missing " + rel)
    if rec.get("sha256") and p.is_file() and file_hash(p) != rec["sha256"]:
        errors.append(name + ": file checksum mismatch")
    receipt = p / "COMPLETE.json" if p.is_dir() else None
    if receipt and receipt.is_file():
        d = read(receipt)
        if rec.get("update") is not None and d.get("update") != rec["update"]:
            errors.append(name + ": wrong checkpoint update")
        if rec.get("identity") and d.get("identity") != rec["identity"]:
            errors.append(name + ": wrong experiment identity")
        for rel, info in d.get("files", {}).items():
            f = contained(p, "/" + rel)
            if not f.is_file():
                errors.append(name + ": missing checkpoint file " + rel)
            elif isinstance(info, dict) and f.stat().st_size != info["size"]:
                errors.append(name + ": checkpoint size mismatch " + rel)
            elif full_hash:
                expected = info["sha256"] if isinstance(info, dict) else info
                if file_hash(f) != expected: errors.append(name + ": checkpoint hash mismatch " + rel)
    return errors

def doctor(p, stage, full_hash=False):
    errors = []
    key = STAGES[stage][0]
    src = contained(p / "fs", RUNS[key]["source"])
    archived = contained(ROOT / "reproduction/fs", RUNS[key]["root"] + "/source_manifest.json")
    for name, expected in read(archived)["files"].items():
        f = src / name
        if not f.is_file() or file_hash(f) != expected: errors.append("Changed staged source: " + name)
    cfg = contained(p / "fs", config_for(stage))
    require_no_kv_training(stage, read(cfg))
    original_cfg = contained(ROOT / "reproduction/fs", config_for(stage))
    if file_hash(cfg) != file_hash(original_cfg): errors.append("Historical configuration changed; make a separately identified new experiment")
    for name, rec in assets_for(p, stage).items():
        errors.extend(check_asset(name, rec, full_hash))
    # Prevent bind-string ambiguities before constructing an Apptainer command.
    for value in [str(p), str(ROOT)] + [r.get("host_path") for r in assets_for(p, stage).values()]:
        if value and any(c in value for c in (":", ",", "\n")):
            errors.append("Paths may not contain colons, commas or newlines: " + value)
    receipt = dict(stage=stage, passed=not errors, errors=errors,
                   tensor_hashes_checked=full_hash, gpu_verified=False,
                   scope="Paths, source provenance and file completeness; no model execution")
    write(p / "logs" / ("doctor-" + stage + ".json"), receipt)
    return receipt

def command(p, stage, image, action="train", engine="apptainer"):
    key, module = STAGES[stage]
    run = RUNS[key]
    config = config_for(stage)
    require_no_kv_training(stage, read(contained(p / "fs", config)))
    if action in {"select", "prune", "verify-export"}:
        if stage != "mask-no-kv": raise ValueError("Use mask-no-kv for selection and physical pruning")
        module = {"select":"vpp2.flexible_mask.select", "prune":"vpp2.flexible_mask.export", "verify-export":"vpp2.flexible_mask.verify_export"}[action]
        extra = []
    elif stage == "export-1500":
        if action == "export":
            module = run["root"] + "/control/export_selected.py"; extra = []
        elif action == "verify":
            module = "vpp2.flexible_mask.verify_export"; extra = []
        else: raise ValueError("Use --action export or verify for export-1500")
    elif action == "export": raise ValueError("Export action requires export-1500")
    elif action == "verify":
        if stage.startswith("recovery"):
            module = "vpp2.flexible_mask.test_gradient_gpu"
            extra = []
        else:
            module = "vpp2.flexible_mask.preflight"
            extra = ["--mode", "probe", "--probe-name", "release_smoke", "--updates", "1"]
    elif action == "resume-check":
        if stage != "recovery-250-to-1000":
            raise ValueError("resume-check belongs to recovery-250-to-1000")
        module = "vpp2.flexible_mask.test_recovery_resume_gpu"
        extra = []
    else:
        extra = []
    args = [engine, "exec", "--nv", "--cleanenv", "--containall",
            "--bind", str(ROOT) + ":/opt/vpp2-release:ro"]
    for prefix in PREFIXES:
        host = contained(p / "fs", prefix)
        host.mkdir(parents=True, exist_ok=True)
        args += ["--bind", str(host) + ":" + prefix + ":rw"]
    for name, rec in assets_for(p, stage).items():
        host = rec.get("host_path")
        if not host: raise ValueError("Missing asset: " + name)
        dest = rec["container_path"]
        # All external assets are read-only; run outputs stay inside the workspace.
        args += ["--bind", str(Path(host).expanduser().resolve()) + ":" + dest + ":ro"]
    env = {
        "PYTHONPATH": run["source"] + "/src",
        "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "VPP2_LOAD_META": "1",
        "WANDB_MODE": "disabled",
        "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", "REQUIRES_SLURM"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "" if action == "export" else "REQUIRES_8_ALLOCATED_GPUS"),
    }
    cfg = read(contained(p / "fs", config))
    if cfg.get("recovery_lr_branch"):
        env["VPP2_CHECKPOINT_LOAD_TMPDIR"] = cfg["recovery_lr_branch"]["load_scratch"]
    for name, value in env.items():
        args += ["--env", name + "=" + value]
    args += [str(Path(image).expanduser().resolve()), "python"]
    if action == "verify-export":
        args += ["-m", module, "--config", config]
    elif stage == "export-1500":
        args += [module] if action == "export" else ["-m",module,"--config",config]
    else:
        args += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8", "-m", module, "--config", config] + extra
    return args

def run_stage(p, stage, image, action, engine):
    d = doctor(p, stage)
    if not d["passed"]: raise ValueError("\n".join(d["errors"]))
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("This preserved eight-GPU recipe requires a Slurm allocation")
    visible = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]
    if action == "verify-export":
        if not visible: raise ValueError("Export verification requires an allocated GPU")
    elif stage == "export-1500":
        if action == "verify" and not visible:raise ValueError("Export verification requires an allocated GPU")
    elif len(visible) != 8:
        raise ValueError("Expected eight Slurm-visible GPUs; physical GPU IDs are never overwritten")
    if not Path(image).is_file(): raise ValueError("Container image not found")
    if not shutil.which(engine): raise ValueError(engine + " is not installed")
    verification = p / "logs" / ("verified-" + stage + ".json")
    if action in {"train", "select", "prune", "verify-export"}:
        if not verification.is_file():
            raise ValueError("Run the GPU verification action in this workspace first")
        v = read(verification)
        if v.get("image_sha256") != file_hash(image) or v.get("stage") != stage or v.get("assets_sha256") != file_hash(p/"assets.json"):
            raise ValueError("Verification belongs to a different container or stage")
        if stage == "recovery-250-to-1000":
            rp=p/"logs/resume-checked.json"
            if not rp.is_file() or read(rp).get("image_sha256") != file_hash(image):
                raise ValueError("Run --action resume-check with this image before the continuation")
    args = command(p, stage, image, action, engine)
    write(p / "logs" / ("launch-" + stage + "-" + action + ".json"),
          dict(argv=args, stage=stage, action=action, job=os.environ["SLURM_JOB_ID"]))
    print(shlex.join(args), flush=True)
    code = subprocess.run(args, check=False).returncode
    if code == 0 and action == "verify":
        write(verification, dict(stage=stage, image_sha256=file_hash(image), assets_sha256=file_hash(p/"assets.json"), passed=True,
              source_manifest_sha256=file_hash(ROOT/"reproduction/release_files.json"),
              scope="Executed smoke/gradient test; not a full GPU replay or simulation evaluation"))
    if code == 0 and action == "resume-check":
        write(p/"logs/resume-checked.json", dict(stage=stage, image_sha256=file_hash(image), passed=True))
    return code

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="Show stages and their exact historical entry points")
    sub.add_parser("verify-source", help="Verify all archived source and evidence hashes")
    sub.add_parser("results", help="Print the recorded checkpoint results")
    q = sub.add_parser("request-stop", help="Ask only the isolated replay to checkpoint and stop")
    q.add_argument("--workspace", required=True);q.add_argument("--stage", choices=STAGES, required=True)
    q.add_argument("--job-id", required=True)
    q = sub.add_parser("init", help="Create an isolated replay workspace; no model weights copied")
    q.add_argument("--workspace", required=True);q.add_argument("--local-assets", action="store_true")
    for name in ("doctor", "command", "run"):
        q = sub.add_parser(name)
        q.add_argument("--workspace", required=True);q.add_argument("--stage", choices=STAGES, required=True)
        if name == "doctor": q.add_argument("--full-hash", action="store_true")
        else:
            q.add_argument("--image", required=True)
            q.add_argument("--action", choices=["verify","resume-check","train","export","select","prune","verify-export"],default="train")
            q.add_argument("--engine",default="apptainer")
    a = p.parse_args(argv)
    try:
        if a.cmd == "list":
            for k, (r,m) in STAGES.items(): print(k + "\n  " + RUNS[r]["source"] + "\n  python -m " + m)
        elif a.cmd == "verify-source":
            result=verify_sources();print(json.dumps(result,indent=2));return 0 if result["passed"] else 1
        elif a.cmd == "results":
            print((ROOT / "results/summary.txt").read_text())
        elif a.cmd == "request-stop":
            if not a.job_id.isdigit():raise ValueError("A numeric Slurm job ID is required")
            w=workspace(a.workspace);key=STAGES[a.stage][0]
            destination=contained(w/"fs",RUNS[key]["root"]+"/control/stop_request.json")
            write(destination,dict(job_id=a.job_id,reason="Standalone replay requested a resumable stop"))
            print(str(destination))
        elif a.cmd == "init": init_workspace(a.workspace,a.local_assets)
        elif a.cmd == "doctor":
            result=doctor(workspace(a.workspace),a.stage,a.full_hash);print(json.dumps(result,indent=2))
            return 0 if result["passed"] else 1
        elif a.cmd == "command":
            print(shlex.join(command(workspace(a.workspace),a.stage,a.image,a.action,a.engine)))
        else:return run_stage(workspace(a.workspace),a.stage,a.image,a.action,a.engine)
    except (ValueError, OSError, KeyError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr);return 2
    return 0
