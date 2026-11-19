"""Frozen spatial visual summaries for selection, never native policy inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from bagel_robotics.data_selection.manifest import digest, file_hash, immutable


def run(config,limit=None):
    from torchvision.models import convnext_tiny
    out=Path(config['curation']['output']);inv=json.loads((out/'inventory.json').read_text())
    cfg=config['curation']['vision'];weights=Path(cfg['weights'])
    weight_hash=file_hash(weights)
    if weight_hash!=cfg['sha256']:raise ValueError('Visual feature weights changed')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(.10)
    torch.manual_seed(config['seed']);torch.set_num_threads(4)
    model=convnext_tiny(weights=None);model.load_state_dict(torch.load(weights,weights_only=True,map_location='cpu'))
    model=model.features.eval().requires_grad_(False).to(device)
    gen=torch.Generator().manual_seed(config['seed'])
    projection=(torch.randn(768,24,generator=gen)/np.sqrt(768)).to(device)
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    output=out/'visual';output.mkdir(exist_ok=True)
    provenance=dict(model='torchvision_convnext_tiny_frozen',weight_sha256=weight_hash,
        input='complete_native_240x416_no_crop',pool=[3,4],channels_per_cell=24,
        projection_seed=config['seed'],history_slots=9,dtype='float16_storage',
        scope='selection-only; learned visual summaries are not verified contact/cue detectors')
    identity=digest(provenance);records=[]
    episodes=[e for e in inv['episodes'] if e['role']!='audit']
    if limit:episodes=episodes[:limit]
    for n,e in enumerate(episodes):
        candidate=out/'candidates'/f"{e['episode']:05d}.npz"
        with np.load(candidate,allow_pickle=False) as z:
            ids=z['needed_frames'];candidate_identity=str(z['identity'])
        expected=digest(dict(vision=identity,candidate=candidate_identity,frames=e['files']['frames.npy']))
        path=output/f"{e['episode']:05d}.npz"
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                if str(z['identity'])!=expected or not np.array_equal(z['frames'],ids):raise ValueError('Stale visual features')
        else:
            frames=np.load(Path(config['demonstrations'])/e['path']/'frames.npy',mmap_mode='r')
            chunks=[]
            with torch.inference_mode():
                for begin in range(0,len(ids),cfg['batch_size']):
                    rgb=torch.from_numpy(np.array(frames[ids[begin:begin+cfg['batch_size']]],copy=True)).to(device).float()/255
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                        raw=model((rgb-mean)/std)
                    pooled=F.adaptive_avg_pool2d(raw.float(),(3,4)).permute(0,2,3,1)
                    feat=(pooled@projection).flatten(1)
                    if not torch.isfinite(feat).all():raise ValueError('Nonfinite visual descriptor')
                    chunks.append(feat.cpu().numpy().astype(np.float16))
            temp=path.with_suffix('.partial')
            with temp.open('wb') as f:np.savez_compressed(f,identity=np.array(expected),frames=ids,features=np.concatenate(chunks))
            temp.replace(path)
        records.append(dict(episode=e['episode'],identity=expected))
        if n%25==0:print(json.dumps(dict(event='frozen_visual_features',done=n+1,total=len(episodes),device=str(device))),flush=True)
    receipt=dict(passed=True,inventory_identity=inv['identity'],identity=identity,provenance=provenance,episodes=records,
        complete=limit is None,native_policy_modified=False,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else 0.)
    immutable(out/('visual_features_smoke.json' if limit else 'visual_features_receipt.json'),receipt)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--limit',type=int)
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.limit)
if __name__=='__main__':main()
