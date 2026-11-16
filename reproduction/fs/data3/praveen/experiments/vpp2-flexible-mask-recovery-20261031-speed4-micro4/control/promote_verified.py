"""Promote unchanged ordinary execution plus an explicitly verified calibration cache release."""
import json,subprocess
from pathlib import Path
from vpp2.flexible_mask.config import load,identity
from vpp2.flexible_mask.continuation import scientific_recipe,validate_parent
from vpp2.flexible_mask.preflight import require_receipt
from vpp2.full_mask.panel import file_hash
from vpp2.compact_kv.spec import atomic_json
root=Path('/data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-speed4-micro4')
old=Path('/data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-speed3-micro4')
c=load(root/'config.json');ident=identity(c);previous=load(old/'config.json')
assert scientific_recipe(c)==scientific_recipe(previous)
manifest=json.loads((root/'source_manifest.json').read_text());old_manifest=json.loads((old/'source_manifest.json').read_text())
changes=sorted(k for k in manifest['files'].keys()|old_manifest['files'].keys() if manifest['files'].get(k)!=old_manifest['files'].get(k))
assert changes==['src/vpp2/flexible_mask/calibration_batch.py','src/vpp2/flexible_mask/train.py'],changes
for name,sha in manifest['files'].items():assert file_hash(Path(manifest['source'])/name)==sha
for name,sha in old_manifest['files'].items():assert file_hash(Path(old_manifest['source'])/name)==sha
comparison=old/'checks/speed_micro4.json';speed=json.loads(comparison.read_text())
assert speed['passed'] and speed['numerics_passed'] and speed['micro_batch']==c['train']['micro_batch']==4
assert speed['execution']==c['execution']
assert speed['identity']==json.loads((old/'inputs.json').read_text())['identity']
for name,sha in speed['evidence'].items():assert file_hash(name)==sha
assert json.loads((old/'checks/speed_micro4_tasks.json').read_text())['passed']
calibration=root/'checks/calibration_equivalence.json';equivalence=json.loads(calibration.read_text())
assert equivalence['passed'] and equivalence['identity']==ident
for name,sha in equivalence['evidence'].items():assert file_hash(name)==sha
ranks=[json.loads((root/'speed_review/micro4/full_calibration'/f'rank{r}.json').read_text()) for r in range(8)]
assert all(x['passed'] and x['identity']==ident and x['frozen_unchanged'] and x['global_batch']==256 and x['micro_batch']==4 for x in ranks)
assert all(len(x['release_statistics'])==8 for x in ranks)
assert all(x['script_sha256']==file_hash(root/'control/check_flexible_calibration_peak.py') for x in ranks)
require_receipt(c,'cpu');parent,receipt=validate_parent(c,'mask');assert receipt['update']==47
report=dict(identity=ident,passed=True,world_size=8,micro_batch=4,global_batch=256,
    benchmark_executed_identity=speed['identity'],calibration_executed_identity=ident,
    inherited_ordinary_execution_changes=changes,ordinary_forward_backward_arithmetic_unchanged=True,
    numerical_comparison_receipt=str(comparison),numerical_comparison_sha256=file_hash(comparison),
    source_manifest_reference=str(old/'source_manifest.json'),source_manifest_reference_sha256=file_hash(old/'source_manifest.json'),
    calibration_equivalence=str(calibration),calibration_equivalence_sha256=file_hash(calibration),
    peak_allocated_gib=max(x['max_allocated_gib'] for x in ranks),
    frozen_reference_unchanged=True,diagnostic_gradient_noninterference=True,contracts=True,
    continuation_checkpoint=c['resume_from'],benchmark_updates_discarded=True,export_equivalence_still_required=True)
atomic_json(root/'checks/gpu.json',report)
(root/'mask/diagnostics').mkdir(parents=True,exist_ok=True)
proof=[]
for src in (parent/'mask/metrics.jsonl',parent/'mask/diagnostics/native_000000.json'):
    dst=root/'mask'/src.relative_to(parent/'mask')
    if dst.exists():assert file_hash(src)==file_hash(dst)
    else:dst.write_bytes(src.read_bytes())
    proof.append(dict(source=str(src),destination=str(dst),sha256=file_hash(dst)))
atomic_json(root/'control/continuation_history.json',dict(parent=str(parent),parent_update=47,files=proof))
atomic_json(root/'control/promotion.json',dict(identity=ident,passed=True,resume_update=47,benchmark_updates_discarded=True,verification=report))
print(json.dumps(dict(identity=ident,passed=True,resume_update=47,peak_allocated_gib=report['peak_allocated_gib'])),flush=True)
