import json
from pathlib import Path
from vpp2.flexible_mask.config import load,identity,sources
from vpp2.flexible_mask.prepare import prepare
from vpp2.flexible_mask.preflight import cpu
root=Path('/data3/praveen/experiments/vpp2-selected-mask-recovery-20261101')
c=load(root/'config.json');prepare(c);cpu(c)
(root/'source_manifest.json').write_text(json.dumps({'identity':identity(c),'files':sources()},indent=2)+'\n')
print(json.dumps({'prepared':True,'identity':identity(c)}),flush=True)
