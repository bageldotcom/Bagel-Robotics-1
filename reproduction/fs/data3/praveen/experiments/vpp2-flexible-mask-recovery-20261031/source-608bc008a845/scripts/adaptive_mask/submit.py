"""Idempotent explicit Slurm submission with afterok handoffs, no cancellation."""
from pathlib import Path
import argparse,fcntl,json,subprocess
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--submit',action='store_true');p.add_argument('--after-job');p.add_argument('--after-job-policy',choices=['afterok','afterany'],default='afterok');a=p.parse_args()
c=json.loads(Path(a.config).read_text());root=Path(c['root'])
from vpp2.adaptive_mask.state import identity
from vpp2.full_mask.panel import immutable
ident=identity(c,json.loads(Path(c['panel_file']).read_text()))
with (root/'control/submission.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 provenance=root/'control/submission_identity.json'
 if a.submit:immutable(provenance,dict(identity=ident))
 elif provenance.exists() and json.loads(provenance.read_text())['identity']!=ident:raise ValueError('Submitted source identity changed')
 receipt=root/'control/submitted_jobs.json'
 if receipt.exists():
  print(receipt.read_text());raise SystemExit(0)
 partial=root/'control/submission_partial.json'
 result=json.loads(partial.read_text()) if partial.exists() else {};previous=a.after_job
 for mode in ('native','distributed','train','audit','final-export'):
  if mode in result:
   previous=result[mode];continue
  relation=a.after_job_policy if mode=='native' and previous==a.after_job else 'afterok'
  cmd=['sbatch','--parsable']+([f'--dependency={relation}:{previous}'] if previous else [])+[str(root/'slurm'/f'{mode}.sbatch')]
  if not a.submit:print(' '.join(cmd));previous=f'<{mode}_job_id>';continue
  job=subprocess.run(cmd,check=True,text=True,capture_output=True).stdout.strip().split(';')[0]
  if not job.isdigit():raise RuntimeError('Unexpected Slurm job id')
  result[mode]=job;previous=job
  # Persist each success: a later failed submission must not duplicate upstream jobs.
  (root/'control/submission_partial.json').write_text(json.dumps(result,indent=2)+'\n')
 if a.submit:
  receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
