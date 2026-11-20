"""DemInf-inspired *episode* ranking. No local score deletes a manipulation stage.

This is an adaptation using frozen spatial vision features and small VAEs, not
a reproduction of DemInf. Observation and action summaries stay separate.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.special import digamma
from scipy.stats import rankdata,spearmanr
import torch
from torch import nn
from bagel_robotics.data_selection.manifest import digest, file_hash, immutable, seed_for
from bagel_robotics.data_selection.descriptors import history_indices


def raw_episode(out,e):
    with np.load(out/'candidates'/f"{e['episode']:05d}.npz",allow_pickle=False) as z:
        c={k:z[k] for k in z.files}
    with np.load(out/'visual'/f"{e['episode']:05d}.npz",allow_pickle=False) as z:
        needed=np.asarray([history_indices(int(t)) for t in c['starts']])
        positions=np.searchsorted(z['frames'],needed)
        if not np.array_equal(z['frames'][positions],needed):raise ValueError('Missing native history feature')
        visual=z['features'][positions].astype(np.float32).reshape(len(needed),-1)
    state=c['state'].copy()
    for start in (3,11):
        q=state[:,start:start+4];norm=np.linalg.norm(q,axis=-1,keepdims=True)
        if not np.isfinite(norm).all() or (norm<1e-8).any():raise ValueError('Invalid state quaternion')
        q/=norm
        q*=np.where(q[np.arange(len(q)),np.abs(q).argmax(-1)]>=0,1.,-1.)[:,None]
    # No action, motion or future-image field enters the observation encoder.
    if not all(np.isfinite(x).all() for x in (visual,state,c['action'],c['motion'])):
        raise ValueError('Nonfinite curation descriptors')
    return c,np.concatenate([visual,state],axis=-1),c['action'].astype(np.float32)


class SummaryVAE(nn.Module):
    def __init__(self,dim,latent):
        super().__init__();self.encoder=nn.Sequential(nn.Linear(dim,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU())
        self.mean=nn.Linear(128,latent);self.logvar=nn.Linear(128,latent)
        self.decoder=nn.Sequential(nn.Linear(latent,128),nn.SiLU(),nn.Linear(128,256),nn.SiLU(),nn.Linear(256,dim))
    def encode(self,x):return self.mean(self.encoder(x))
    def forward(self,x,sample=True):
        h=self.encoder(x);mu=self.mean(h);lv=self.logvar(h).clamp(-8,4)
        z=mu+torch.randn_like(mu)*(.5*lv).exp() if sample else mu
        return self.decoder(z),mu,(-.5*(1+lv-mu.square()-lv.exp())).mean()


def reconstruction(pred,target,state_block=False):
    sq=(pred-target).square()
    if state_block:return .5*(sq[:,:-16].mean()+sq[:,-16:].mean())
    return sq.mean()


def normalized(x,mean,scale):return np.clip((x-mean)/scale,-6,6).astype(np.float32)


def balanced_tasks(step,batch,tasks,rng):
    # A continuous task stream; no task receives a permanent remainder advantage.
    order=(np.arange(batch)+(step-1)*batch)%tasks
    rng.shuffle(order)
    return order


def ksg_cross_episode(obs,act,eps,reference,ks=(5,6,7)):
    """KSG-style score with different-episode neighbours and effective-N term.

    Same-episode exclusion is our conservative extension. It prevents adjacent
    windows from manufacturing support; scores are used as within-task ranks.
    """
    from scipy.spatial.distance import cdist
    out=[]
    ro=obs[reference];ra=act[reference];re=eps[reference]
    for begin in range(0,len(obs),128):
        os=cdist(obs[begin:begin+128],ro);ac=cdist(act[begin:begin+128],ra)
        valid=eps[begin:begin+128,None]!=re[None,:]
        if (valid.sum(-1)<max(ks)).any():raise ValueError('Insufficient independent-episode neighbours')
        joint=np.maximum(os,ac);joint[~valid]=np.inf
        rhos=np.partition(joint,np.array(ks)-1,axis=1)[:,np.array(ks)-1]
        terms=[]
        for j,k in enumerate(ks):
            ns=((os<rhos[:,j,None])&valid).sum(-1)
            na=((ac<rhos[:,j,None])&valid).sum(-1)
            terms.append(digamma(k)+digamma(valid.sum(-1)+1)-digamma(ns+1)-digamma(na+1))
        out.append(np.mean(terms,axis=0))
    return np.concatenate(out)


def run(config):
    out=Path(config['curation']['output']);cfg=config['curation']['quality']
    inv=json.loads((out/'inventory.json').read_text());vision=json.loads((out/'visual_features_receipt.json').read_text())
    if not vision['complete'] or vision['inventory_identity']!=inv['identity']:raise ValueError('Incomplete visual stage')
    if (out/'quality_receipt.json').exists():
        r=json.loads((out/'quality_receipt.json').read_text())
        if (r['inventory_identity']!=inv['identity'] or r['config']!=cfg or
            r['vision_identity']!=vision['identity'] or r['model_sha256']!=file_hash(out/'quality_models.pt')):
            raise ValueError('Stale quality receipt')
        return r
    torch.set_num_threads(4);torch.manual_seed(config['seed']);np.random.seed(config['seed'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.10)
    torch.use_deterministic_algorithms(True)
    fitted=[];held=[];input_hashes={}
    for n,e in enumerate(inv['episodes']):
        if e['role']=='audit':continue
        input_hashes[str(e['episode'])]={name:file_hash(out/name/f"{e['episode']:05d}.npz") for name in ('candidates','visual')}
        c,o,a=raw_episode(out,e);ix=c['score_indices']
        (fitted if e['role']=='fit' else held).append((e,o[ix],a[ix]))
    xo=np.concatenate([v[1] for v in fitted]);xa=np.concatenate([v[2] for v in fitted])
    vo=np.concatenate([v[1] for v in held]);va=np.concatenate([v[2] for v in held])
    means=[x.mean(0) for x in (xo,xa)];scales=[np.maximum(x.std(0),1e-3) for x in (xo,xa)]
    xo=normalized(xo,*[v[0] for v in (means,scales)]);xa=normalized(xa,*[v[1] for v in (means,scales)])
    vo=normalized(vo,means[0],scales[0]);va=normalized(va,means[1],scales[1])
    ids=defaultdict(list);offset=0
    for e,o,a in fitted:ids[e['task']].extend(range(offset,offset+len(o)));offset+=len(o)
    ids={k:np.array(v) for k,v in ids.items()};tasks=sorted(ids)
    mo=SummaryVAE(xo.shape[1],cfg['obs_latent']).to(device);ma=SummaryVAE(xa.shape[1],cfg['action_latent']).to(device)
    opt=torch.optim.AdamW(list(mo.parameters())+list(ma.parameters()),lr=cfg['learning_rate'],weight_decay=1e-4)
    rng=np.random.default_rng(config['seed']);best=float('inf');stale=0;history=[]
    path=out/'quality_models.pt'
    def evaluate():
        losses=[]
        with torch.no_grad():
            for begin in range(0,len(vo),512):
                o=torch.from_numpy(vo[begin:begin+512]).to(device);a=torch.from_numpy(va[begin:begin+512]).to(device)
                po,_,_=mo(o,False);pa,_,_=ma(a,False)
                losses.append((len(o),float(reconstruction(po,o,True)),float(reconstruction(pa,a))))
        return [sum(n*x[i] for n,*x in losses)/sum(x[0] for x in losses) for i in (0,1)]
    baseline=[float(.5*(np.square(vo[:,:-16]).mean()+np.square(vo[:,-16:]).mean())),float(np.square(va).mean())]
    for step in range(1,cfg['max_updates']+1):
        task_order=balanced_tasks(step,256,len(tasks),rng)
        ix=np.array([rng.choice(ids[tasks[t]]) for t in task_order])
        o=torch.from_numpy(xo[ix]).to(device);a=torch.from_numpy(xa[ix]).to(device)
        po,_,ko=mo(o);pa,_,ka=ma(a)
        lo=reconstruction(po,o,True);la=reconstruction(pa,a);loss=lo+la+1e-3*(ko+ka)
        if not torch.isfinite(loss):raise FloatingPointError('Quality helper diverged')
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(list(mo.parameters())+list(ma.parameters()),1.);opt.step()
        if step%cfg['check_every']==0:
            ev=evaluate();total=sum(v/b for v,b in zip(ev,baseline));improved=total<best*.995
            history.append(dict(update=step,train_observation=float(lo.detach()),train_action=float(la.detach()),
                                monitor_observation=ev[0],monitor_action=ev[1],relative_total=total))
            if total<best:
                best=total;temp=path.with_suffix('.partial');torch.save(dict(observation=mo.state_dict(),action=ma.state_dict(),
                    mean=means,scale=scales,update=step,monitor=ev,baseline=baseline,config=cfg,
                    inventory_identity=inv['identity'],vision_identity=vision['identity']),temp);temp.replace(path)
            stale=0 if improved else stale+1
            (out/'quality_training_progress.json').write_text(json.dumps(history,indent=2)+'\n')
            print(json.dumps(dict(event='quality_helpers',**history[-1],stale_checks=stale)),flush=True)
            if step>=400 and stale>=cfg['patience']:break
    best_state=torch.load(path,map_location='cpu',weights_only=False)
    mo.load_state_dict(best_state['observation']);ma.load_state_dict(best_state['action']);mo.eval();ma.eval()
    output=out/'summaries';output.mkdir(exist_ok=True);score_groups=defaultdict(list)
    for e in inv['episodes']:
        if e['role']=='audit':continue
        c,o,a=raw_episode(out,e);zo=[];za=[]
        with torch.no_grad():
            for begin in range(0,len(o),512):
                zo.append(mo.encode(torch.from_numpy(normalized(o[begin:begin+512],means[0],scales[0])).to(device)).cpu().numpy())
                za.append(ma.encode(torch.from_numpy(normalized(a[begin:begin+512],means[1],scales[1])).to(device)).cpu().numpy())
        zo=np.concatenate(zo);za=np.concatenate(za)
        # Diverse selection also retains uncompressed physical-action descriptors.
        temp=output/f"{e['episode']:05d}.partial"
        with temp.open('wb') as f:np.savez_compressed(f,observation=zo,action=za,starts=c['starts'])
        temp.replace(output/f"{e['episode']:05d}.npz")
        if e['role']=='fit':score_groups[e['task']].append((e,c['score_indices'],zo[c['score_indices']],za[c['score_indices']]))
    quality={};stabilities={}
    reconstruction_pass=all(v<b*.90 for v,b in zip(best_state['monitor'],baseline))
    for task,parts in sorted(score_groups.items()):
        obs=np.concatenate([p[2] for p in parts]);act=np.concatenate([p[3] for p in parts]);eps=np.concatenate([np.full(len(p[1]),p[0]['episode']) for p in parts])
        # Whiten each space on fitting episodes only, preserving their separate metrics.
        obs=(obs-obs.mean(0))/np.maximum(obs.std(0),1e-3);act=(act-act.mean(0))/np.maximum(act.std(0),1e-3)
        per_episode=sorted(set(eps.tolist()));rep_scores=[]
        for repeat in range(cfg['repeats']):
            rg=np.random.default_rng(seed_for(config['seed'],'quality-reference',task,repeat))
            ref=np.concatenate([rg.choice(np.flatnonzero(eps==ep),min(8,int((eps==ep).sum())),replace=False) for ep in per_episode])
            # Tiny deterministic jitter breaks numerical distance ties only in scoring.
            score=ksg_cross_episode(obs+rg.normal(0,1e-7,obs.shape),act+rg.normal(0,1e-7,act.shape),eps,ref,cfg['neighbors'])
            score=np.clip(score,*np.quantile(score,[.01,.99]))
            rep_scores.append([float(score[eps==ep].mean()) for ep in per_episode])
        rep_scores=np.array(rep_scores);avg=rep_scores.mean(0)
        corrs=[float(spearmanr(rep_scores[i],rep_scores[j]).statistic) for i in range(len(rep_scores)) for j in range(i)]
        stable=bool(corrs and np.isfinite(corrs).all() and np.median(corrs)>=cfg['minimum_rank_correlation'])
        ranks=(rankdata(avg)-1)/max(1,len(avg)-1)
        for j,ep in enumerate(per_episode):quality[str(ep)]=dict(task=task,episode=ep,rank=float(ranks[j]),score=float(avg[j]),repeat_std=float(rep_scores[:,j].std()),usable=stable and reconstruction_pass)
        stabilities[task]=dict(median_rank_agreement=float(np.nanmedian(corrs)) if np.isfinite(corrs).any() else None,
                              ranking_usable=stable and reconstruction_pass)
        print(json.dumps(dict(event='episode_quality',task=task,**stabilities[task])),flush=True)
    receipt=dict(passed=True,inventory_identity=inv['identity'],vision_identity=vision['identity'],config=cfg,
        model_sha256=file_hash(path),input_cache_hashes=input_hashes,selected_update=best_state['update'],training_history=history,
        reconstruction_check_passed=reconstruction_pass,task_checks=stabilities,episodes=quality,
        audit_episodes_used=0,observation_future_leakage=False,
        limitations=['Episode quality is a weak tie-breaker, not verified success or a deletion rule.',
            'Frozen spatial vision and native-history summaries differ from the original DemInf implementation.',
            'No individual low-score contact or memory window is removed by this scorer.'])
    immutable(out/'quality_receipt.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(json.loads(Path(a.config).read_text()))
if __name__=='__main__':main()
