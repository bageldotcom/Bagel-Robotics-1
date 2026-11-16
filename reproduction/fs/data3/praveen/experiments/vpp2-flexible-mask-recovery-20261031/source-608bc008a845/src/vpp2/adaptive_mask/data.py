"""Keep future scoring targets separate from policy inputs and conditioning cache."""
from contextlib import contextmanager
import torch
from vpp2.full_mask.data import WindowReader,collate,prepared as native_prepared
from vpp2.full_mask.panel import seed_for

@contextmanager
def raw_reader(reader,future=False):
    old=(reader.runtime_cache,reader.native_cache,reader.read_future_rgb)
    try:
        reader.runtime_cache=None;reader.native_cache={};reader.read_future_rgb=future
        yield
    finally:reader.runtime_cache,reader.native_cache,reader.read_future_rgb=old

@torch.no_grad()
def prepare(model,reader,rows,positions,seed,strata):
    parts=native_prepared(model,collate(reader,rows,positions),seed)
    return action_draw(model,parts,positions,seed,strata)

@torch.no_grad()
def action_draw(model,parts,positions,seed,strata):
    result=list(parts);action=parts[0]['action'];scheduler=model.train_action_scheduler
    noises=[];times=[]
    for p,s in zip(positions,strata):
        gen=torch.Generator(device=action.device).manual_seed(seed_for(seed,'adaptive-action',int(p),int(s)))
        noises.append(torch.randn(action.shape[1:],generator=gen,device=action.device,dtype=action.dtype))
        u=(int(s)+torch.rand((),generator=gen,device=action.device))/4
        times.append((scheduler._phi(u,scheduler.shift)*scheduler.num_train_timesteps).to(action.dtype))
    eps=torch.stack(noises);time=torch.stack(times)
    result[3]=scheduler.add_noise(action,eps,time);result[4]=time;result[5]=scheduler.training_target(action,eps,time)
    return tuple(result)

def future_mask(rows,device):
    return torch.tensor([[r['frame']+16<r['length'],r['frame']+32<r['length']] for r in rows],device=device,dtype=torch.float32)

@torch.no_grad()
def future_targets(model,reader,rows):
    # Targets returned separately; never inserted into prepared policy inputs.
    with raw_reader(reader,future=True):
        sample=collate(reader,rows,[-1-r['id'] for r in rows])
    video=sample['video'].to(device=model.device,dtype=model.torch_dtype)
    latents=model._encode_video_latents(video,tiled=False)
    if latents.shape[1:]!=(16,5,30,52):raise ValueError('Future target geometry changed')
    return latents.detach()
