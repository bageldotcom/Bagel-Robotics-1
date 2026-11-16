"""Independent checks and fail-closed readiness for curated calibration data."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from .curation import load_panel,history_indices,SCHEMA,fingerprint_window
from .panel import file_hash,immutable,BalancedStream,digest


def verify_native_contract(config):
    import torch
    path=Path(config['teacher_bundle'])/'action.pt'
    checkpoint=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    train=checkpoint['resolved_config']['data']['train'];processor=train['processor']
    expected={'action_horizon':32,'action_video_freq_ratio':4,'condition_history_frames':8,
              'condition_history_stride':25,'condition_include_episode_first':True,
              'condition_history_target_prefix':True,'video_size':[240,416]}
    for k,v in expected.items():
        if train[k]!=v:raise ValueError(f'Native input contract changed: {k}')
    required={'num_image_obs_steps':17,'use_stepwise_action_norm':False,
              'norm_default_mode':'z-score','norm_exception_mode':None,
              'action_output_dim':16,'proprio_output_dim':16}
    for k,v in required.items():
        if processor[k]!=v:raise ValueError(f'Native processor changed: {k}')
    return dict(checkpoint=str(path),checkpoint_step=checkpoint['step'],**expected,processor=required)


def verify(config,panel,check_reader=True):
    if panel['schema'] not in (SCHEMA,'robodojo-curated-calibration-v3','robodojo-critical-calibration-v1'):raise ValueError('Unexpected curated panel schema')
    if panel['schema'] in ('robodojo-curated-calibration-v3','robodojo-critical-calibration-v1'):
        from .curation_v3_checks import verify_roles
        verify_roles(config,panel)
    groups=defaultdict(list);roles=defaultdict(set);ids=set();fps=defaultdict(set)
    source=Path(config['demonstrations']);source_manifest=json.loads((source/'manifest.json').read_text())
    by_episode={int(e['id']):e for e in source_manifest['episodes']}
    for r in panel['windows']:
        if r['id'] in ids:raise ValueError('Duplicate row id')
        ids.add(r['id']);roles[r['role']].add(r['episode']);groups[(r['role'],r['episode'])].append(r)
        fps[r['role']].add(r['content_sha256'])
        if r['frame']<0 or r['frame']+32>r['length']:raise ValueError('Incomplete actions')
        if r['history_indices']!=history_indices(r['frame']):raise ValueError('Native history mismatch')
        if r['action_interval']!=[r['frame'],r['frame']+32]:raise ValueError('Interval mismatch')
        e=by_episode[r['episode']]
        if e['split']!='train' or e['task']!=r['task'] or e['path']!=r['episode_path']:raise ValueError('Source identity mismatch')
        if r['source_files']!=e['files']:raise ValueError('Source hashes mismatch')
    if roles['selection_initial']&roles['monitor']:raise ValueError('Episode leakage')
    if set(panel['official_validation_episodes'])&set.union(*roles.values()):raise ValueError('Final-validation leakage')
    if fps['selection_initial']&fps['monitor']:raise ValueError('Exact content leakage')
    for key,rs in groups.items():
        ts=sorted(r['frame'] for r in rs)
        if any(b-a<32 for a,b in zip(ts,ts[1:])):raise ValueError('Overlapping selected actions')
        if len(rs)>config['curation']['max_windows_per_episode']:raise ValueError('Per-episode cap exceeded')
    # Verify every selected native window, not only stale metadata in a manifest.
    for ep in sorted(set.union(*roles.values())):
        e=by_episode[ep]
        for name in ('actions.npy','states.npy'):
            if file_hash(source/e['path']/name)!=e['files'][name]:raise ValueError('Source action/state bytes changed')
    for index,r in enumerate(panel['windows'],1):
        if fingerprint_window(config,r)!=r['content_sha256']:raise ValueError('Selected native window content changed')
        if index%2000==0 or index==len(panel['windows']):
            print(json.dumps(dict(event='verified_native_window_bytes',done=index,total=len(panel['windows']))),flush=True)
    for role in roles:
        if len(fps[role])!=sum(r['role']==role for r in panel['windows']):raise ValueError('Exact duplicate within a panel role')
    native_contract=verify_native_contract(config)
    current=json.loads((source/'dataset_stats.json').read_text())
    teacher=json.loads((Path(config['teacher_bundle'])/'dataset_stats.json').read_text())
    if current!=teacher or file_hash(source/'dataset_stats.json')!=panel['normalization_sha256']:raise ValueError('Normalization drift')
    selection=[r for r in panel['windows'] if r['role']=='selection_initial']
    stream=BalancedStream(selection,config['seed']);resumed=BalancedStream(selection,config['seed'])
    order=[stream.at(i)['id'] for i in range(4096)]
    if order[517:2089]!=[resumed.at(i)['id'] for i in range(517,2089)]:raise ValueError('Resume order changed')
    if sorted(order[:256])!=sorted(stream.at(i)['id'] for rank in range(8) for i in range(rank,256,8)):raise ValueError('Rank partition changes sampling')
    counts=Counter(stream.at(i)['task'] for i in range(35*40))
    if set(counts.values())!={40}:raise ValueError('Task weighting is not uniform')
    native_reads=[]
    if check_reader:
        import torch
        from .data import WindowReader
        reader=WindowReader(config)
        reader.runtime_cache=None  # Audit source observations, never cached encoder tensors.
        for task in panel['tasks']:
            rs=[r for r in selection if r['task']==task]
            for r in [min(rs,key=lambda r:r['frame']),max(rs,key=lambda r:r['frame'])]:
                sample=reader.read(r,1234)
                arrays=source/r['episode_path'];a=np.load(arrays/'actions.npy',mmap_mode='r');s=np.load(arrays/'states.npy',mmap_mode='r');f=np.load(arrays/'frames.npy',mmap_mode='r')
                expected=torch.from_numpy(np.array(f[history_indices(r['frame'])],copy=True)).float().div(255).mul(2).sub(1).permute(1,0,2,3)
                if not torch.equal(sample['condition_video'],expected):raise ValueError('RGB history differs')
                if not torch.equal(sample['action'],reader.normalizers['action'].forward(torch.tensor(a[r['frame']:r['frame']+32].copy()))):raise ValueError('Action normalization differs')
                if not torch.equal(sample['proprio'],reader.normalizers['state'].forward(torch.tensor(s[r['frame']:r['frame']+1].copy()))):raise ValueError('State normalization differs')
                if sample['action_is_pad'].any():raise ValueError('Unexpected action padding')
                if sample['video'].shape!=(3,17,240,416):raise ValueError('Video geometry')
                if not reader.read_future_rgb and sample['video'][:,9:].count_nonzero():raise ValueError('Unexpected future RGB read')
                native_reads.append([r['episode'],r['frame']])
    source_code=Path(__file__).resolve().parents[3]
    ident=digest(dict(config=config,panel=panel['identity'],source={str(p.relative_to(source_code)):file_hash(p) for p in sorted((source_code/'src/vpp2').rglob('*.py'))}))
    report=dict(passed=True,identity=ident,panel_identity=panel['identity'],window_counts=panel['counts'],tasks=len(panel['tasks']),
                selected_episodes=len(roles['selection_initial']),monitor_episodes=len(roles['monitor']),
                action_overlap_count=0,episode_leakage_count=0,exact_cross_split_duplicates=0,
                equal_task_weighting=True,resume_and_rank_order=True,normalization_unchanged=True,
                loader_checks=native_reads,native_checkpoint_contract=native_contract,selected_window_hashes_verified=len(panel['windows']),
                model_forward_equivalence='pending_gpu_check',
                semantic_review='pending',mask_learning_ready=False)
    return report


def require_training_readiness(config,panel):
    root=Path(config.get('readiness_dir',config['curation']['output']))
    required=['cpu_checks.json','visual_review_receipt.json','model_probe_receipt.json']
    for name in required:
        path=Path(config.get('readiness_receipts',{}).get(name,root/name))
        if not path.exists():raise RuntimeError(f'Curated mask learning requires {path}')
        d=json.loads(path.read_text())
        if d.get('panel_identity')!=panel['identity'] or d.get('passed') is not True:
            raise RuntimeError(f'Curated readiness receipt did not pass: {path}')
        if name=='visual_review_receipt.json' and config.get('review_protocol')=='staged_visual_v1':
            from vpp2.critical_windows.staged_review import assess
            review_config=config
            if config.get('visual_review_config'):
                path=Path(config['visual_review_config'])
                if file_hash(path)!=config['visual_review_config_sha256']:raise RuntimeError('Visual review configuration changed')
                review_config=json.loads(path.read_text())
            current,_=assess(review_config,panel)
            if not current['passed'] or current['identity']!=d.get('identity'):
                raise RuntimeError('Staged review or its evidence changed after publication')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--skip-reader',action='store_true');a=p.parse_args()
    c=json.loads(Path(a.config).read_text());panel=load_panel(c['panel_file']);report=verify(c,panel,not a.skip_reader)
    out=(Path(c['root'])/'checks/structure_checks.json') if a.skip_reader else Path(c.get('cpu_receipt_file',Path(c.get('readiness_dir',c['curation']['output']))/'cpu_checks.json'))
    immutable(out,report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
