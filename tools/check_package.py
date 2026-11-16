"""Run lightweight package checks without importing training models or allocating GPUs."""
from pathlib import Path
import json
import os
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from release_tools.cli import verify_sources

def main():
    checks={}
    source=verify_sources();checks["source_hashes"]=source
    test=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],cwd=ROOT,text=True,capture_output=True)
    checks["cpu_tests"]={"passed":test.returncode==0,"output":test.stdout+test.stderr}
    shell=subprocess.run(["bash","-n",str(ROOT/"scripts/train.sbatch")],text=True,capture_output=True)
    checks["slurm_syntax"]={"passed":shell.returncode==0,"stderr":shell.stderr}
    missing=[]
    for p in [ROOT/"README.md"]+list((ROOT/"docs").glob("*.md")):
        for dest in re.findall(r"\]\(([^)]+)\)",p.read_text()):
            if dest.startswith(("https:","http:","#")):continue
            if not (p.parent/dest.split("#")[0]).exists():missing.append(str(p.relative_to(ROOT))+":"+dest)
    checks["documentation_links"]={"passed":not missing,"missing":missing}
    bad=[]
    for p in ROOT.rglob("*"):
        if "dist" not in p.parts and p.is_file() and p.suffix in (".pt",".safetensors",".bin"):
            bad.append(str(p.relative_to(ROOT)))
    checks["weight_free"]={"passed":not bad,"unexpected_tensors":bad}
    checks["passed"]=all(v["passed"] for k,v in checks.items() if isinstance(v,dict))
    checks["gpu_replay_run"]=False
    checks["container_built"]=False
    checks["full_tensor_hash_scan"]=False
    checks["training_or_slurm_jobs_changed"]=False
    checks["scope"]="Standalone packaging and existing results; not new GPU training or evaluation"
    out=ROOT/"verification";out.mkdir(exist_ok=True)
    (out/"packaging_checks.json").write_text(json.dumps(checks,indent=2)+"\n")
    print(json.dumps({k:v for k,v in checks.items() if k!="cpu_tests"},indent=2))
    print(checks["cpu_tests"]["output"])
    return 0 if checks["passed"] else 1
if __name__=="__main__":raise SystemExit(main())
