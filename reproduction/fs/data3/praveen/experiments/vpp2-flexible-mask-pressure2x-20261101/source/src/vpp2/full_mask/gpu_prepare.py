"""Scheduled diagnostics and deterministic conditioning export; never policy training."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import torch
import numpy as np
from .curation import load_panel
from .data import WindowReader,collate,prepared
from .panel import immutable,file_hash,digest,seed_for
from .runtime import initialize,install,cache,predict,model_versions,check_frozen,verify_gpu
from vpp2.compact_kv.losses import action_error


def checkpoint_identity(config):
    bundle=Path(config['teacher_bundle'])
    return {name:file_hash(bundle/name) for name in ('video.pt','action.pt','dataset_stats.json')}


def export_inputs(config,panel):
    model,reader,_=initialize(config);versions=model_versions(model)
    root=Path(config['curation']['output'])/'native_conditioning';root.mkdir(exist_ok=True)
    checkpoints=checkpoint_identity(config)
    identity=digest(dict(panel=panel['identity'],checkpoints=checkpoints,preprocessing='native_build_inputs_no_future_v2'))
    rows=[];parity=[]
    for i,r in enumerate(panel['windows']):
        path=root/(r['window_id']+'.pt')
        sample=collate(reader,[r],[i])
        with torch.no_grad():inputs=model.build_inputs(sample,training_stage='joint',encode_video_target=False)
        payload={k:(v.detach().cpu() if torch.is_tensor(v) else v) for k,v in inputs.items()}
        if payload['input_latents'] is not None:raise AssertionError('Future video was encoded')
        data=dict(identity=identity,window_id=r['window_id'],inputs=payload)
        if not path.exists():
            temp=path.with_suffix('.partial');torch.save(data,temp);temp.replace(path)
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['identity']!=identity or saved['window_id']!=r['window_id']:raise ValueError('Stale native input cache')
        for k,v in payload.items():
            if torch.is_tensor(v):
                if not torch.equal(v,saved['inputs'][k]):raise AssertionError(f'Cached native input differs: {k}')
            elif v!=saved['inputs'][k]:raise AssertionError('Native scalar differs')
        if i<2:
            from vpp2.compact_kv.model import prepare_batch,native_cache,action_prediction
            with torch.no_grad():
                live=prepare_batch(model,sample,config['seed'])
                loaded={k:(v.to(model.device) if torch.is_tensor(v) else v) for k,v in saved['inputs'].items()}
                reused=prepare_batch(model,sample,config['seed'],prepared_inputs=loaded)
                a=action_prediction(model,live[0],native_cache(model,live[0],live[1],live[2],live[6]),live[3],live[4],live[6])
                b=action_prediction(model,reused[0],native_cache(model,reused[0],reused[1],reused[2],reused[6]),reused[3],reused[4],reused[6])
                if not torch.equal(a,b):raise AssertionError('Native cached/online policy parity failed')
                parity.append(r['window_id'])
        rows.append(dict(window_id=r['window_id'],path=str(path),sha256=file_hash(path)))
        if (i+1)%100==0:print(json.dumps(dict(event='native_input_export',done=i+1,total=len(panel['windows']))),flush=True)
    check_frozen(model,versions)
    manifest=dict(schema='native-conditioning-cache-v2',panel_identity=panel['identity'],identity=identity,
                  checkpoints=checkpoints,rows=rows,passed=True,bitwise_policy_parity=parity,
                  future_video_encoded=False,source_identity=panel['source_identity'])
    immutable(root/'manifest.json',manifest)


def probe(config,panel,windows_per_role=16,draws=2):
    model,reader,controller=initialize(config);branch=install(model);versions=model_versions(model)
    root=Path(config['curation']['output']);reserve=[json.loads(x) for x in (root/'reserve_manifest.jsonl').read_text().splitlines()]
    by_task=defaultdict(lambda:defaultdict(list))
    for r in panel['windows']:
        if r['role']=='selection_initial':by_task[r['task']]['selection'].append(r)
    for r in reserve:
        if not r['shares_selected_episode']:by_task[r['task']]['reserve'].append(r)
    result=[];profiles={}
    for task in panel['tasks']:
        for role in ('selection','reserve'):
            candidates=by_task[task][role];rng=np.random.default_rng(seed_for(config['seed'],'model-probe',task,role))
            order=rng.permutation(len(candidates));chosen=[];episodes=set()
            for j in order:
                r=candidates[int(j)]
                if r['episode'] in episodes:continue
                chosen.append(r);episodes.add(r['episode'])
                if len(chosen)==windows_per_role:break
            if len(chosen)!=windows_per_role:raise ValueError('Insufficient disjoint probe episodes')
            total_h=torch.zeros_like(controller.hidden_scores,dtype=torch.float32,device='cpu')
            total_f=torch.zeros_like(controller.ffn_scores,dtype=torch.float32,device='cpu')
            for i,r in enumerate(chosen):
                for draw in range(draws):
                    # Same strata distribution in both groups, different reproducible noise per window.
                    position=-3_000_000-int(seed_for(config['seed'],r['window_id'],draw)%1_000_000_000)
                    parts=prepared(model,collate(reader,[r],[position]),config['seed'],[(i+2*draw)%4])
                    h=torch.ones_like(controller.hidden_scores,requires_grad=True)
                    f=torch.ones_like(controller.ffn_scores,requires_grad=True)
                    pred=predict(model,parts,cache(branch,parts,(h,f)))
                    loss=action_error(pred,parts[5],parts[0]['action_loss_is_pad']).mean()
                    gh,gf=torch.autograd.grad(loss,(h,f))
                    if not torch.isfinite(gh).all() or not torch.isfinite(gf).all():raise FloatingPointError('Nonfinite gate sensitivity')
                    total_h+=gh.detach().float().abs().cpu();total_f+=gf.detach().float().abs().cpu()
                    result.append(dict(task=task,role=role,episode=r['episode'],frame=r['frame'],draw=draw,gt_action_error=float(loss.detach())))
                    del pred,loss,gh,gf,h,f,parts
            profiles[(task,role)]=(total_h/(len(chosen)*draws),total_f/(len(chosen)*draws))
            print(json.dumps(dict(event='action_sensitivity_probe',task=task,role=role,windows=len(chosen))),flush=True)
    check_frozen(model,versions)
    comparisons={}
    def agreement(a,b):
        k=max(1,int(a.numel()*.05));ia=set(torch.argsort(a,descending=True,stable=True)[:k].tolist());ib=set(torch.argsort(b,descending=True,stable=True)[:k].tolist())
        return len(ia&ib)/len(ia|ib)
    for task in panel['tasks']:
        a,af=profiles[(task,'selection')];b,bf=profiles[(task,'reserve')]
        comparisons[task]=dict(hidden_top5_jaccard=agreement(a,b),ffn_top5_jaccard=[agreement(x,y) for x,y in zip(af,bf)])
    packed={f'{task}/{role}/{kind}':v for (task,role),values in profiles.items() for kind,v in zip(('hidden','ffn'),values)}
    torch.save(packed,root/'model_probe_profiles.pt')
    receipt=dict(passed=True,panel_identity=panel['identity'],checkpoints=checkpoint_identity(config),
                 windows_per_role=windows_per_role,draws=draws,rows=result,comparisons=comparisons,
                 frozen_native_weights=True,optimizer_steps=0,policy_retention_established=False,
                 interpretation='Gradient agreement is a local coverage diagnostic; it does not establish policy retention.')
    immutable(root/'model_probe_receipt.json',receipt)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--mode',choices=['preflight','probe','cache'],required=True)
    p.add_argument('--windows-per-role',type=int,default=16);p.add_argument('--draws',type=int,default=2);a=p.parse_args()
    config=json.loads(Path(a.config).read_text());panel=load_panel(config['panel_file'])
    if not torch.cuda.is_available():raise RuntimeError('Use an allocated Slurm GPU')
    torch.cuda.set_device(0)
    if a.mode=='preflight':verify_gpu(config,panel,0,1)
    elif a.mode=='probe':
        if panel['schema'] in ('robodojo-curated-calibration-v3','robodojo-critical-calibration-v1'):
            from .curation_v3_probe import probe as probe_v3
            probe_v3(config,panel,a.windows_per_role,a.draws)
        else:probe(config,panel,a.windows_per_role,a.draws)
    else:export_inputs(config,panel)


if __name__=='__main__':main()
