from pathlib import Path
import argparse,hashlib,json
import torch
from vpp2.adaptive_mask.state import identity
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--gpus',type=int,required=True);p.add_argument('--mode',required=True);a=p.parse_args()
c=json.loads(Path(a.config).read_text());root=Path(c['root']);manifest=json.loads((root/'source_manifest.json').read_text())
for name,expected in manifest['files'].items():
 if hashlib.sha256((Path(manifest['source'])/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Frozen source changed: '+name)
if Path(__file__).resolve().parents[2]!=Path(manifest['source']).resolve():raise RuntimeError('Launcher/source manifest mismatch')
panel=json.loads(Path(c['panel_file']).read_text());ident=identity(c,panel)
required=['cpu']+(['native'] if a.mode in ('distributed','train','audit','final-export') else [])+(['distributed'] if a.mode in ('train','audit','final-export') else [])
for key in required:
 receipt=json.loads((root/'checks'/f'{key}.json').read_text())
 if receipt.get('passed') is not True or receipt.get('identity')!=ident:raise RuntimeError('Missing or stale readiness: '+key)
if torch.cuda.device_count()!=a.gpus:raise RuntimeError('Slurm GPU allocation mismatch')
for i in range(a.gpus):
 free,total=torch.cuda.mem_get_info(i)
 if free<68*2**30:raise RuntimeError(f'Allocated GPU {i}: only {free/2**30:.1f} GiB free')
print('Frozen artifacts, checks and allocated GPU memory verified',flush=True)
