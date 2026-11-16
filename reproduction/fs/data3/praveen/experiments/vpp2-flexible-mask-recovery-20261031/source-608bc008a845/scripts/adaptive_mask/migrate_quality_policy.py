"""Document an authorized acceptance-policy revision without changing learned state."""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import torch
from migrate_execution_resume import equal
from vpp2.adaptive_mask.quality_policy import ACTION_ONLY, LEGACY, policy_mode
from vpp2.adaptive_mask.state import identity, save


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--work',required=True)
    args=parser.parse_args();config=json.loads(Path(args.config).read_text());root=Path(config['root']);work=Path(args.work)
    previous=json.loads((work/'before_revision/config.json').read_text())
    if policy_mode(previous)!=LEGACY or policy_mode(config)!=ACTION_ONLY:raise ValueError('Unexpected policy transition')
    reduced=copy.deepcopy(config);reduced.pop('quality_gate')
    equal(previous,reduced,'unchanged_config_except_acceptance_policy')
    old_manifest=json.loads((work/'before_revision/source_manifest.json').read_text())
    manifest=json.loads((root/'source_manifest.json').read_text())
    old_files=old_manifest['files'];new_files=manifest['files']
    changed={name for name in set(old_files)|set(new_files) if name.startswith('src/') and old_files.get(name)!=new_files.get(name)}
    allowed={'src/vpp2/adaptive_mask/'+name+'.py' for name in ['quality_policy','monitor','quality','preflight','train','audit','report']}
    allowed.add('src/vpp2/full_mask/conditioning_cache.py')
    if changed!=allowed:raise ValueError(('Unexpected production changes',changed ^ allowed))
    for name in new_files:
        if sha(Path(manifest['source'])/name)!=new_files[name]:raise ValueError('Published source changed: '+name)
    panel=json.loads(Path(config['panel_file']).read_text());new_identity=identity(config,panel)
    for kind in ('cpu','native','distributed'):
        receipt=json.loads((root/f'checks/{kind}.json').read_text())
        if not receipt.get('passed') or receipt['identity']!=new_identity:raise ValueError('Missing current verification: '+kind)
    preserved=json.loads((work/'preservation.json').read_text());original=Path(preserved['checkpoint'])
    if sha(original)!=preserved['sha256']:raise ValueError('Preserved state changed')
    old=torch.load(original,map_location='cpu',weights_only=False)
    current=torch.load(root/'runs/latest.pt',map_location='cpu',weights_only=False)
    if current['identity']==new_identity and (work/'migration.json').exists():
        receipt=json.loads((work/'migration.json').read_text())
        if receipt['new_identity']!=new_identity or current['budget']['attempt']<receipt['attempt']:raise ValueError('Inconsistent prior migration')
        print('Acceptance-policy migration already recorded; preserving continued state.');return
    equal(old,current,'current_vs_preserved_checkpoint')
    if old.get('quality_gate_mode',LEGACY)!=LEGACY or old.get('finished') or old.get('terminated'):
        raise ValueError('Expected an active legacy-policy checkpoint')
    revision=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        kind='scientific_acceptance_policy_revision',old_policy=LEGACY,new_policy=ACTION_ONLY,
        old_identity=old['identity'],new_identity=new_identity,source=manifest['source'],source_sha256=manifest['sha256'],
        effective_after_attempt=old['budget']['attempt'],preserved_checkpoint_sha256=preserved['sha256'],
        authorization='User: please go ahead as per your recommendation, after the action-only gate proposal.',
        unchanged=['losses','reliability','action_thresholds','confidence_bounds','frozen_weights','data_order','randomness','optimizer','mask_scores','capacity_state'],
        video_diagnostics_retained=True,historical_results_rewritten=False,
        stability_policy='Require five consecutive checks under one acceptance policy; preserve earlier history as recorded.')
    new=copy.deepcopy(old);new['identity']=new_identity;new['quality_gate_mode']=ACTION_ONLY
    new['protocol_revisions']=old.get('protocol_revisions',[])+[revision]
    for key in old:
        if key not in ('identity','quality_gate_mode','protocol_revisions'):equal(old[key],new[key],key)
    migrated=work/'migrated_checkpoint.pt';save(migrated,new)
    restored=torch.load(migrated,map_location='cpu',weights_only=False);equal(new,restored,'serialized_migration')
    save(root/'runs/latest.pt',restored)
    receipt=dict(**revision,attempt=old['budget']['attempt'],lineage_update=old['state']['lineage_update'],
        cursor=old['state']['cursor'],sample_noise_optimizer_masks_rng_coverage_history_unchanged=True,
        tracking_run_id=new.get('tracking_run_id',old['identity'][:12]),checkpoint=str(migrated),checkpoint_sha256=sha(migrated))
    (work/'migration.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
