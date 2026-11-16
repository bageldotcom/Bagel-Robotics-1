"""Adopt resident teacher storage only after exact state parity and measured gain."""
import argparse,json,statistics
from pathlib import Path
from .config import load,identity
from vpp2.compact_kv.spec import atomic_json
from vpp2.full_mask.panel import file_hash


def recipe(config):
    c=json.loads(json.dumps(config))
    for key in ('root','execution','performance_reference_root'):c.pop(key,None)
    c['wandb'].pop('name',None)
    return c


def compare(config):
    root=Path(config['root']);base=Path(config['performance_reference_root'])
    baseline=load(base/'config.json')
    if recipe(config)!=recipe(baseline):raise ValueError('Performance comparison changed the scientific recipe')
    if json.loads((root/'source_manifest.json').read_text())['files']!=json.loads((base/'source_manifest.json').read_text())['files']:
        raise ValueError('Performance comparison requires identical source bytes')
    def records(folder,ident):
        p=folder/'gpu_probe_uninterrupted';out=[]
        for rank in range(8):
            item=json.loads((p/f'fingerprint_rank{rank}.json').read_text())
            if item['identity']!=ident or item['update']!=3:raise ValueError('Incomplete performance run')
            out.append(item)
        metrics=[json.loads(line) for line in (p/'metrics.jsonl').read_text().splitlines()]
        if [m['update'] for m in metrics]!=[1,2,3]:raise ValueError('Performance updates are not unique')
        return out,metrics
    left,lm=records(base,identity(baseline));right,rm=records(root,identity(config))
    failures=[]
    for rank,(a,b) in enumerate(zip(left,right)):
        for key in a['fingerprint']:
            if a['fingerprint'][key]!=b['fingerprint'][key]:failures.append(f'rank{rank}/{key}')
    numerical=('gt_action','teacher_video_raw','teacher_video_gated','late_kv_raw','late_kv_gated','reliability','total','video_gradient_norm','gradient/hidden','gradient/ffn','gradient/cross')
    for a,b in zip(lm,rm):
        for key in numerical:
            if a[key]!=b[key]:failures.append(f'update{a["update"]}/{key}')
    baseline_seconds=statistics.mean(m['seconds_per_update'] for m in lm[1:])
    candidate_seconds=statistics.mean(m['seconds_per_update'] for m in rm[1:])
    memory=max(r['peak_allocated_gib'] for r in right)
    speedup=baseline_seconds/candidate_seconds
    report=dict(identity=identity(config),passed=not failures and speedup>=1.05 and memory<=76.,
        exact_state_and_loss_parity=not failures,mismatches=failures,
        baseline_identity=identity(baseline),baseline_root=str(base),
        baseline_seconds_per_update=baseline_seconds,candidate_seconds_per_update=candidate_seconds,
        speedup=speedup,minimum_speedup=1.05,max_allocated_gib_all_ranks=memory,allocated_limit_gib=76.,
        baseline_metrics_sha256=file_hash(base/'gpu_probe_uninterrupted/metrics.jsonl'),candidate_metrics_sha256=file_hash(root/'gpu_probe_uninterrupted/metrics.jsonl'))
    atomic_json(root/'checks/performance.json',report)
    print(json.dumps(report),flush=True)
    if not report['passed']:raise RuntimeError('Resident teacher not accepted; inspect performance receipt')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();compare(load(a.config))
if __name__=='__main__':main()
