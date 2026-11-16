"""Preserve all scientific state across a proven execution-only source revision."""
import argparse,copy,datetime,hashlib,json
from pathlib import Path
import numpy as np
import torch
from vpp2.adaptive_mask.state import identity,save


def equal(left,right,path='checkpoint'):
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or left.dtype!=right.dtype or not torch.equal(left,right):raise AssertionError(path)
    elif isinstance(left,np.ndarray):
        if not isinstance(right,np.ndarray) or left.dtype!=right.dtype or not np.array_equal(left,right):raise AssertionError(path)
    elif isinstance(left,dict):
        if left.keys()!=right.keys():raise AssertionError(path+' keys')
        for k in left:equal(left[k],right[k],path+'.'+str(k))
    elif isinstance(left,(list,tuple)):
        if type(left)!=type(right) or len(left)!=len(right):raise AssertionError(path+' sequence')
        for i,(a,b) in enumerate(zip(left,right)):equal(a,b,path+'.'+str(i))
    elif left!=right:raise AssertionError(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--work',required=True);a=p.parse_args()
    config=json.loads(Path(a.config).read_text());root=Path(config['root']);work=Path(a.work)
    oldconfig=json.loads((work/'before_optimization/config.json').read_text());normalized=copy.deepcopy(config)
    for key in ('cache_hard_masks','mixed_attention_checkpointing'):normalized.get('execution',{}).pop(key,None)
    equal(oldconfig,normalized,'scientific_config')
    oldmanifest=json.loads((work/'before_optimization/source_manifest.json').read_text());manifest=json.loads((root/'source_manifest.json').read_text())
    changed=[name for name,h in oldmanifest['files'].items() if manifest['files'].get(name)!=h and name.startswith('src/')]
    allowed={'src/vpp2/adaptive_mask/controller.py','src/vpp2/adaptive_mask/execution.py','src/vpp2/adaptive_mask/runtime.py','src/vpp2/adaptive_mask/train.py','src/vpp2/full_mask/conditioning_cache.py'}
    if set(changed)!=allowed:raise ValueError(('Unexpected source changes',changed))
    if any(name.startswith('src/') and name not in oldmanifest['files'] for name in manifest['files']):raise ValueError('Unexpected new production module')
    bench=json.loads((work/'benchmark.json').read_text())
    if not bench.get('passed'):raise ValueError('Paired execution benchmark incomplete')
    case=next(x for x in bench['candidates'] if x['name']==bench['selected'])
    if case['micro']!=config['train']['micro_batch'] or case['layers']!=config['execution']['video_checkpoint_layers']:raise ValueError('Untested batching or checkpoint pattern')
    if case['cache']!=config['execution']['cache_hard_masks'] or case['mixed']!=config['execution']['mixed_attention_checkpointing']:raise ValueError('Untested execution flags')
    if not case.get('equivalent') or not bench['stress']['candidate'].get('equivalent'):raise ValueError('Execution equivalence failed')
    for name in ('controller.py','execution.py','runtime.py'):
        relative=Path('src/vpp2/adaptive_mask')/name
        if (Path(manifest['source'])/relative).read_bytes()!=(work/'benchmark_source'/relative).read_bytes():raise ValueError('GPU-tested core code changed')
    panel=json.loads(Path(config['panel_file']).read_text());newid=identity(config,panel)
    for kind in ('cpu','native','distributed'):
        checked=json.loads((root/f'checks/{kind}.json').read_text())
        if not checked.get('passed') or checked['identity']!=newid:raise ValueError('Unverified final source: '+kind)
    provenance=json.loads((work/'preservation.json').read_text());original=Path(provenance['checkpoint'])
    if hashlib.sha256(original.read_bytes()).hexdigest()!=provenance['sha256']:raise ValueError('Preserved checkpoint changed')
    old=torch.load(original,map_location='cpu',weights_only=False);current=torch.load(root/'runs/latest.pt',map_location='cpu',weights_only=False)
    if current['identity']==newid and (work/'migration.json').exists():
        receipt=json.loads((work/'migration.json').read_text())
        if receipt['new_identity']!=newid:raise ValueError('Migration receipt differs')
        print('Execution migration already complete; preserving newer resumed state.',flush=True);return
    equal(old,current,'current_vs_preserved')
    if old['identity']!=bench['checkpoint_identity'] or hashlib.sha256(original.read_bytes()).hexdigest()!=bench['checkpoint_sha256']:raise ValueError('Wrong benchmark starting state')
    revision=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),old_identity=old['identity'],new_identity=newid,
                  source=str(manifest['source']),source_sha256=manifest['sha256'],benchmark_sha256=hashlib.sha256((work/'benchmark.json').read_bytes()).hexdigest(),
                  benchmark_selected=bench['selected'],preserved_checkpoint_sha256=provenance['sha256'])
    new=copy.deepcopy(old);new['identity']=newid;new['tracking_run_id']=old.get('tracking_run_id',old['identity'][:12]);new['execution_revisions']=old.get('execution_revisions',[])+[revision]
    for key in old:
        if key not in ('identity','tracking_run_id','execution_revisions'):equal(old[key],new[key],key)
    migrated=work/'migrated_checkpoint.pt';save(migrated,new)
    verified=torch.load(migrated,map_location='cpu',weights_only=False);equal(new,verified,'serialized_state')
    save(root/'runs/latest.pt',verified)
    receipt=dict(**revision,attempt=old['budget']['attempt'],lineage_update=old['state']['lineage_update'],cursor=old['state']['cursor'],
                 sample_noise_optimizer_masks_rng_coverage_gate_history_unchanged=True,tracking_run_id=new['tracking_run_id'],checkpoint=str(migrated),
                 checkpoint_sha256=hashlib.sha256(migrated.read_bytes()).hexdigest())
    (work/'migration.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
