"""Predeclared held-out audit, run only after an immutable final selection.

Audit outcomes never enter the score optimizer, capacity controller or candidate
selection. Offline audit success is not a claim of closed-loop policy retention.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path

from vpp2.full_mask.panel import digest, file_hash, immutable, seed_for


def build_manifest(config, panel, source):
    """Metadata-only temporal coverage; no model output or demonstration scoring."""
    forbidden = {r['episode'] for r in panel['windows']} | set(panel['official_validation_episodes'])
    reserved = sorted((r for r in panel['episode_partitions'] if r['role'] == 'audit'),
                      key=lambda r: (r['task'], r['episode']))
    if len(reserved) != 280 or Counter(r['task'] for r in reserved) != Counter({t: 8 for t in panel['tasks']}):
        raise ValueError('Expected eight reserved audit episodes per task')
    by_episode = {int(e['id']): e for e in source['episodes']}
    suites = {r['task']: r['suite'] for r in panel['windows']}
    first_id = max(r['id'] for r in panel['windows']) + 1
    rows = []
    for order, entry in enumerate(reserved):
        ep = entry['episode']; source_row = by_episode[ep]
        if ep in forbidden or source_row['task'] != entry['task'] or source_row['split'] != 'train':
            raise ValueError('Audit episode leakage or wrong source')
        last = source_row['length'] - 33
        # Four complete, disjoint future/action intervals. Short episodes are an
        # explicit failure to review, not silently sampled with duplicate windows.
        if last < 3 * 33:
            raise ValueError('Audit episode too short for four disjoint windows')
        frames = [round(last * i / 3) for i in range(4)]
        for quarter, frame in enumerate(frames):
            rows.append(dict(id=first_id + len(rows), task=entry['task'], task_id=panel['tasks'].index(entry['task']),
                suite=suites[entry['task']], episode=ep, episode_path=source_row['path'], length=source_row['length'],
                role='audit', frame=frame, temporal_quarter=quarter, audit_stratum=quarter, owner_index=order,
                history_indices=[0]+[max(0,frame+offset) for offset in range(-175,1,25)],
                source_files=source_row['files'], text=source_row['text'], text_sha256=source_row['text_sha256'],
                prompt=source_row['prompt'], critical_annotation_scope='not_annotated', phase_status='unverified',
                selection='predeclared_uniform_temporal_coverage', action_interval=[frame,frame+32]))
    record=dict(schema='adaptive-mask-heldout-audit-v1', panel_identity=panel['identity'], seed=config['seed'],
        source_manifest_sha256=config['artifacts']['source_manifest']['sha256'], tasks=panel['tasks'],
        rows=rows, episodes=len(reserved), windows=len(rows), native_action_windows=len(reserved),
        independence_scope='Episode-disjoint from fitting and selection in this experiment; prior cross-experiment use not established.',
        selection_feedback_allowed=False, simulation=False)
    record['identity']=digest(record)
    validate_manifest(record,panel)
    return record


def validate_manifest(manifest,panel):
    if digest({k:v for k,v in manifest.items() if k!='identity'}) != manifest['identity']:
        raise ValueError('Audit manifest content changed')
    if manifest['panel_identity'] != panel['identity'] or manifest['selection_feedback_allowed']:
        raise ValueError('Wrong audit provenance or feedback policy')
    expected={r['episode'] for r in panel['episode_partitions'] if r['role']=='audit'}
    forbidden={r['episode'] for r in panel['windows']} | set(panel['official_validation_episodes'])
    rows=manifest['rows']; ids={r['id'] for r in panel['windows']}
    if {r['episode'] for r in rows} != expected or expected & forbidden:
        raise ValueError('Audit episodes overlap fitting/selection/official validation')
    if len({r['id'] for r in rows}) != len(rows) or ids & {r['id'] for r in rows}:
        raise ValueError('Audit/cache window identities collide')
    if any(r['role']!='audit' or r['frame']+32>=r['length'] for r in rows):
        raise ValueError('Invalid audit row')
    if Counter(r['episode'] for r in rows) != Counter({ep:4 for ep in expected}):
        raise ValueError('Audit episode coverage incomplete')
    return True


def run(config,panel):
    import torch
    from .state import identity
    from .quality_policy import policy_mode, LEGACY
    from .runtime import distributed,initialize,barrier,check_frozen,model_versions
    from .monitor import evaluate,native_evaluate
    from .preflight import validate_inputs
    rank,world=distributed()
    root=Path(config['root']); final_path=root/'final_selection.json'
    if not final_path.exists():
        if rank==0:immutable(root/'audit/not_run.json',dict(reason='No finalized selection; audit remains sealed.',evaluated=False))
        barrier();return
    validate_inputs(config,panel)
    ident=identity(config,panel); final=json.loads(final_path.read_text()); final_hash=file_hash(final_path)
    from vpp2.critical_windows.inference import model_identity
    hashes=[model_identity(config) if rank==0 else None]
    if torch.distributed.is_initialized():torch.distributed.broadcast_object_list(hashes,src=0)
    native_receipt=json.loads((root/'checks/native.json').read_text())
    if native_receipt['identity']!=ident or native_receipt['checkpoint_hashes']!=hashes[0]:
        raise ValueError('Teacher checkpoint changed since native preflight')
    if final.get('quality_gate_mode',LEGACY)!=policy_mode(config):
        raise ValueError('Final selection quality policy differs from audit policy')
    if final['identity']!=ident or not final['stable']:
        raise ValueError('Audit requires a finalized selection from this exact experiment')
    artifact=config['artifacts']['audit_manifest']
    if file_hash(artifact['path'])!=artifact['sha256'] or final['audit_manifest_sha256']!=artifact['sha256']:
        raise ValueError('Audit manifest was not pinned before final selection')
    manifest=json.loads(Path(artifact['path']).read_text());validate_manifest(manifest,panel)
    output=root/'audit/report.json'
    if output.exists():
        old=json.loads(output.read_text())
        if old['identity']!=ident or old['final_selection_sha256']!=final_hash:
            raise ValueError('Refusing to audit a second selection against the same held-out data')
        return
    # Also pin a started receipt: a failed job can retry this candidate, never a new one.
    if rank==0:immutable(root/'audit/started.json',dict(identity=ident,final_selection_sha256=final_hash,
        audit_manifest_sha256=artifact['sha256']))
    barrier()
    audit_config=copy.deepcopy(config);audit_config['runtime_conditioning_cache']=None
    model,reader,controller,branch,inventory=initialize(audit_config);versions=model_versions(model)
    controller.set_selection(final['selection'])
    if inventory.count(controller.selection())!=final['parameters']:
        raise ValueError('Final audit parameter count mismatch')
    # Open only the explicitly pinned audit episodes in this independent process.
    # Training readers retain their original exclusion rule.
    reader.allowed_episodes={r['episode'] for r in manifest['rows']}
    reader.runtime_cache=None;reader.native_cache={};model._full_mask_conditioning_cache=None
    selected=copy.deepcopy(controller.state_dict())
    verify=reader.read;seen=set()
    def audit_read(row,position):
        if row['role']!='audit' or row['episode'] not in reader.allowed_episodes:
            raise ValueError('Non-audit input in audit reader')
        if row['episode'] not in seen:
            for name,expected in row['source_files'].items():
                if file_hash(Path(config['demonstrations'])/row['episode_path']/name)!=expected:
                    raise ValueError('Audit source changed')
            seen.add(row['episode'])
        return verify(row,position)
    reader.read=audit_read
    rows=manifest['rows'];entries=[]
    for ep in sorted(reader.allowed_episodes):
        group=[r for r in rows if r['episode']==ep]
        entries.append(group[seed_for(config['seed'],'audit-native-quarter',ep)%len(group)])
    monitor=evaluate(config,panel,model,reader,branch,controller,0,rank,world,full=True,rows_override=rows)
    native=native_evaluate(config,panel,model,branch,controller,rank,world,entries_override=entries)
    check_frozen(model,versions)
    from .physical import native_compact_audit
    physical_native=native_compact_audit(config,panel,model,controller,inventory,rank,world,entries,native)
    check_frozen(model,versions)
    current=controller.state_dict()
    for name,value in selected.items():
        if torch.is_tensor(value):
            if not torch.equal(value,current[name]):raise AssertionError('Audit changed mask scores')
        elif value!=current[name]:raise AssertionError('Audit changed channel identities')
    if file_hash(final_path)!=final_hash:raise ValueError('Final candidate changed during audit')
    if rank==0:
        report=dict(identity=ident,final_selection_sha256=final_hash,audit_manifest_sha256=artifact['sha256'],
            evaluated=True,monitor=monitor,native=physical_native,masked_native=native,episodes=len(reader.allowed_episodes),windows=len(rows),
            native_action_windows=len(entries),physically_compact_native_evaluated=True,
            offline_criteria_passed=monitor['passed'] and native['passed'] and physical_native['passed'],
            quality_gate_mode=policy_mode(config),video_diagnostic_passed=monitor['video_quality']['passed'],
            protocol_revisions=final.get('protocol_revisions',[]),
            policy_retention_established=False,selection_feedback_allowed=False,
            independence_scope=manifest['independence_scope'],simulation=False)
        immutable(output,report)
        from .report import plot_monitor
        plot_monitor(monitor,physical_native,[],controller.selection(),final['attempt'],root/'audit')
        print(json.dumps({k:report[k] for k in ('episodes','windows','native_action_windows','offline_criteria_passed')}),flush=True)
    barrier()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--prepare',action='store_true');a=p.parse_args()
    c=json.loads(Path(a.config).read_text());panel=json.loads(Path(c['panel_file']).read_text())
    if a.prepare:
        source=json.loads(Path(c['artifacts']['source_manifest']['path']).read_text())
        result=build_manifest(c,panel,source);path=Path(c['root'])/'audit/manifest.json'
        immutable(path,result);print(json.dumps(dict(path=str(path),sha256=file_hash(path),episodes=result['episodes'],windows=result['windows'])))
    else:run(c,panel)

if __name__=='__main__':main()
