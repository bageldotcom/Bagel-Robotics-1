import json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from vpp2.flexible_mask.config import load,identity
from vpp2.flexible_mask.preflight import require_receipt
from vpp2.flexible_mask.export import write_export
from vpp2.compact_kv.model import native_config
from vpp2.compact_kv.spec import atomic_json
from vpp2.full_mask.panel import file_hash,digest
root=Path('/data3/praveen/experiments/vpp2-selected-mask-recovery-20261101')
c=load(root/'config.json');require_receipt(c,'cpu')
selection=Path(c['recovery_handoff']['selection_root']);rp=selection/'reports/comparison.json'
r=json.loads(rp.read_text());p=json.loads((selection/'protocol.json').read_text())
if not r['coverage_complete'] or r['protocol_identity']!=c['recovery_handoff']['protocol_identity']:
    raise ValueError('Incomplete or wrong selection comparison')
if file_hash(selection/'selection_rule.json')!=c['recovery_handoff']['selection_rule_sha256']:
    raise ValueError('Selection rule changed')
if r['ranking_rule']!=json.loads((selection/'selection_rule.json').read_text()):raise ValueError('Wrong ranking rule')
chosen=r['recommended_checkpoint'];u=r['recommended_update']
if chosen!=next(x for x in p['candidates'] if x['update']==u):raise ValueError('Wrong comparison candidate')
source=selection/f'exports/update_{u:06d}';receipt=json.loads((source/'COMPLETE.json').read_text())
if receipt['checkpoint_identity']!=chosen['checkpoint_identity']:raise ValueError('Wrong trained checkpoint')
for name,item in receipt['files'].items():
    if file_hash(source/name)!=item['sha256']:raise ValueError('Changed candidate '+name)
mp=torch.load(source/'masks.pt',map_location='cpu',weights_only=True)
if mp['checkpoint_identity']!=chosen['checkpoint_identity'] or digest(mp['selection'])!=chosen['mask_identity']:
    raise ValueError('Mask and video do not belong together')
payload=torch.load(source/'video.pt',map_location='cpu',mmap=True,weights_only=True)
if payload['checkpoint_identity']!=chosen['checkpoint_identity']:raise ValueError('Wrong trained weights')
kwargs=OmegaConf.to_container(native_config(c).model.video_dit_config,resolve=True)
print(json.dumps({'selected_update':u,'parameters':chosen['parameters'],'physical_export_started':True}),flush=True)
path=write_export(c,payload['dit'],mp['selection'],kwargs,receipt,u)
manifest=json.loads((path/'manifest.json').read_text())
if manifest['parameters']!=chosen['parameters']:raise ValueError('Physical model size differs from selected candidate')
atomic_json(root/'selected.json',dict(identity=identity(c),status='selected_for_early_recovery',update=u,
    checkpoint_identity=chosen['checkpoint_identity'],mask_identity=chosen['mask_identity'],parameters=chosen['parameters'],
    selection_report=str(rp),selection_report_sha256=file_hash(rp),source_export=str(source),
    mask_convergence_claimed=False,policy_retention_established=False,cross_environment_checks=False))
atomic_json(root/'export_pending_verification.json',dict(path=str(path),identity=identity(c),verified=False))
print(json.dumps({'export_complete':str(path),'selected_update':u,'parameters':manifest['parameters']}),flush=True)
