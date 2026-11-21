"""Rank at most two ready checkpoints using full native action diagnostics."""
import argparse,json
from pathlib import Path
from bagel_robotics.adaptive_pruning.config import load, identity
from bagel_robotics.adaptive_pruning.runtime import Runtime, distributed
from bagel_robotics._internal.training.checkpoint import load as restore
from bagel_robotics.adaptive_pruning.diagnose import native
from bagel_robotics.adaptive_pruning.selection import candidate_key
from bagel_robotics._internal.compact_kv.spec import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();c=load(a.config)
    rank,_,_=distributed(c);root=Path(c['root']);ident=identity(c)
    status=json.loads((root/'mask/status.json').read_text())
    if status['status']!='ready_for_recovery' or status['identity']!=ident:
        raise RuntimeError('Mask learning has not converged; no automatic export')
    runtime=Runtime(c);state=restore(runtime,root/'mask',ident,'mask')
    candidates=[x for x in state['history'] if x.get('readiness',{}).get('status')=='ready_for_recovery'
                and x['parameters']<c['original_video_parameters']]
    candidates=candidates[-2:]
    if not candidates:raise RuntimeError('No smaller, qualified checkpoint')
    panel=json.loads(Path(c['panel_file']).read_text());results=[]
    for candidate in candidates:
        tag=f"update_{candidate['update']:06d}"
        state=restore(runtime,root/'mask',ident,'mask',tag=tag)
        if runtime.masks.selection()[0]!=candidate['selection']:raise ValueError('Candidate masks do not match checkpoint')
        result=native(runtime,panel,candidate['update'],full=True)
        receipt=json.loads((root/'mask/checkpoints'/tag/'COMPLETE.json').read_text())
        results.append(dict(candidate,full_native_actions=result['actions'],checkpoint=tag,
                            checkpoint_identity=receipt['checkpoint_identity']))
        if rank==0:atomic_json(root/f"mask/diagnostics/finalist_{candidate['update']:06d}.json",result)
    winner=min(results,key=candidate_key)
    if rank==0:atomic_json(root/'selected.json',dict(identity=ident,status='ready_for_export',candidate=winner,
        finalists=[dict(update=x['update'],rank_key=candidate_key(x)) for x in results],
        pre_recovery_teacher_closeness_required=False,policy_retention_established=False))

if __name__=='__main__':main()
