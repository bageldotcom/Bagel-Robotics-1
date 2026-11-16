"""Compare the full 256-window calibration measurement at identical weights."""
import argparse,json,math
from pathlib import Path
from vpp2.flexible_mask.config import load,identity
from vpp2.compact_kv.spec import atomic_json
from vpp2.full_mask.panel import file_hash
p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);a=p.parse_args()
root=Path('/data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-speed4-micro4')
old=Path('/data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-speed2')
c=load(root/'config.json');checks=[];evidence={}
for rank in range(8):
    paths=[old/'speed_review/optimized/full_calibration'/f'rank{rank}.json',root/'speed_review'/a.candidate/'full_calibration'/f'rank{rank}.json']
    previous,current=[json.loads(p.read_text()) for p in paths]
    assert previous['passed'] and current['passed'] and current['identity']==identity(c)
    assert previous['parent_update']==current['parent_update']==47
    assert previous['calibration_position_update']==current['calibration_position_update']==193
    assert previous['global_batch']==current['global_batch']==256
    for key,values in previous['gradient_norms'].items():
        for family,x,y in zip(('hidden','ffn','cross'),values,current['gradient_norms'][key]):
            rel=abs(x-y)/max(abs(x),1e-10)
            checks.append(dict(rank=rank,objective=key,family=family,baseline=x,candidate=y,relative_difference=rel))
    evidence.update({str(p):file_hash(p) for p in paths})
max_relative=max(x['relative_difference'] for x in checks)
report=dict(identity=identity(c),passed=math.isfinite(max_relative) and max_relative<=.01,
    relative_tolerance=.01,maximum_relative_difference=max_relative,checks=checks,evidence=evidence,
    same_weights_same_256_windows=True,gradient_vectors_averaged_before_norm=True,script_sha256=file_hash(__file__))
atomic_json(root/'checks/calibration_equivalence.json',report)
print(json.dumps({k:report[k] for k in ('passed','maximum_relative_difference','relative_tolerance')}),flush=True)
if not report['passed']:raise RuntimeError('Calibration measurement changed excessively')
