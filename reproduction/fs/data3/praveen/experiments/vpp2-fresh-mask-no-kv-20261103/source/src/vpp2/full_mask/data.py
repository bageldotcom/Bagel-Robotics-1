"""Reuse native decoded observation cache without decoding or encoding future video."""
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
import torch
from .panel import file_hash, seed_for


class WindowReader:
    def __init__(self, config):
        self.root=Path(config['demonstrations']); self.cache=OrderedDict(); self.text={}
        self.allowed_episodes=None
        self.runtime_cache=None
        if config.get('runtime_conditioning_cache'):
            from .conditioning_cache import ConditioningCache
            self.runtime_cache=ConditioningCache(config)
        if config.get('curation',{}).get('schema')=='robodojo-curated-calibration-v3':
            inv=json.loads((Path(config.get('curation_artifacts',config['curation']['output']))/'inventory.json').read_text())
            self.allowed_episodes={e['episode'] for e in inv['episodes'] if e['role']!='audit'}
        self.stats=json.loads((self.root/'dataset_stats.json').read_text())
        teacher=json.loads((Path(config['teacher_bundle'])/'dataset_stats.json').read_text())
        if self.stats!=teacher: raise ValueError('Normalization must match original checkpoint')
        from vpp2.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer
        self.normalizers={}
        for field in ('action','state'):
            stats={k.removeprefix('global_'):torch.tensor(v,dtype=torch.float32)
                   for k,v in self.stats[field]['default'].items() if k.startswith('global_')}
            self.normalizers[field]=SingleFieldLinearNormalizer(stats,mode='z-score')
        self.read_future_rgb=bool(config.get('curation',{}).get('read_future_rgb',False))
        self.native_cache={}
        cache_path=config.get('native_conditioning_manifest')
        if cache_path:
            index=json.loads(Path(cache_path).read_text())
            expected=config.get('native_conditioning_manifest_sha256')
            if not expected or file_hash(Path(cache_path))!=expected:raise ValueError('Unpinned native conditioning cache')
            if index.get('passed') is not True or not index.get('bitwise_policy_parity'):raise ValueError('Unverified native cache')
            panel=json.loads(Path(config['panel_file']).read_text())
            if index['panel_identity']!=panel['identity']:raise ValueError('Native cache panel mismatch')
            self.native_cache={r['window_id']:r for r in index['rows']}
            self.native_cache_identity=index['identity']
    def norm(self, x, field):
        return self.normalizers[field].forward(torch.as_tensor(x,dtype=torch.float32))
    def read(self, row, position):
        if self.allowed_episodes is not None and row['episode'] not in self.allowed_episodes:
            raise ValueError('Final-audit episode is sealed; cannot enter training or development reader')
        if self.runtime_cache is not None:
            inputs=self.runtime_cache.load(row['id'])
            if inputs is not None:
                return dict(_native_prepared=inputs,pilot_position=torch.tensor(position))
        if self.native_cache:
            entry=self.native_cache[row['window_id']];p=Path(entry['path'])
            if file_hash(p)!=entry['sha256']:raise ValueError('Native input cache changed')
            saved=torch.load(p,map_location='cpu',weights_only=True)
            if saved['identity']!=self.native_cache_identity or saved['window_id']!=row['window_id']:
                raise ValueError('Native input identity changed')
            return dict(_native_prepared=saved['inputs'],pilot_position=torch.tensor(position))
        ep=row['episode']
        if ep not in self.cache:
            arrays={name:np.load(self.root/row['episode_path']/f'{name}.npy',mmap_mode='r',allow_pickle=False)
                    for name in ('frames','actions','states')}
            if arrays['frames'].shape!=(row['length'],3,240,416) or arrays['frames'].dtype!=np.uint8:
                raise ValueError('Unexpected decoded RGB format')
            if any(arrays[k].shape!=(row['length'],16) for k in ('actions','states')):
                raise ValueError('Unexpected action/state format')
            self.cache[ep]=arrays
            if len(self.cache)>8:self.cache.popitem(last=False)
        self.cache.move_to_end(ep); arrays=self.cache[ep]; frame=row['frame']
        if not 0<=frame<=row['length']-32:raise ValueError('Calibration requires 32 valid actions')
        if row['text'] not in self.text:
            p=self.root/row['text']
            if file_hash(p)!=row['text_sha256']:raise ValueError('Native text changed')
            t=torch.load(p,map_location='cpu',weights_only=True)
            if t['prompt']!=row['prompt'] or tuple(t['context'].shape)!=(128,4096) or not t['mask'].all():
                raise ValueError('Native context mismatch')
            self.text[row['text']]=t
        text=self.text[row['text']]
        history=[0]+[max(0,frame+o) for o in range(-175,1,25)]
        if 'history_indices' in row and history!=row['history_indices']:
            raise ValueError('Manifest changed the native history')
        future=np.minimum(frame+np.arange(4,33,4),row['length']-1).tolist()
        # Only the independently supplied nine history frames are model inputs.
        # The 17-frame video satisfies the native shape contract; prepare_batch
        # explicitly skips its target encoding for these action/KV objectives.
        indices=history+future if self.read_future_rgb else history
        rgb=torch.from_numpy(np.array(arrays['frames'][indices],copy=True)).float().div(255).mul(2).sub(1).permute(1,0,2,3)
        video=rgb if self.read_future_rgb else torch.cat([rgb,torch.zeros(3,8,240,416,dtype=rgb.dtype)],dim=1)
        action=self.norm(np.array(arrays['actions'][frame:frame+32],copy=True),'action')
        state=self.norm(np.array(arrays['states'][frame:frame+1],copy=True),'state')
        return dict(_conditioning_key=torch.tensor(row['id']),video=video,video_num_frames=torch.tensor(17),condition_video=video[:,:9].clone(),
                    action=action,proprio=state,action_is_pad=torch.zeros(32,dtype=torch.bool),
                    image_is_pad=torch.tensor([False]*9+[frame+int(o)>=row['length'] for o in range(4,33,4)]),context=text['context'],
                    context_mask=text['mask'].bool(),pilot_position=torch.tensor(position))


def collate(reader, rows, positions):
    samples=[reader.read(r,p) for r,p in zip(rows,positions)]
    if all('_native_prepared' in s for s in samples):
        inputs={}
        for k,v in samples[0]['_native_prepared'].items():
            values=[s['_native_prepared'][k] for s in samples]
            if torch.is_tensor(v):inputs[k]=torch.cat(values,dim=0)
            else:
                if any(x!=v for x in values):raise ValueError('Mixed native scalar values')
                inputs[k]=v
        return dict(_native_prepared=inputs,pilot_position=torch.stack([s['pilot_position'] for s in samples]))
    if any('_native_prepared' in s for s in samples):raise ValueError('Partially cached batch')
    return {k:torch.stack([s[k] for s in samples]) for k in samples[0]}


def prepared(reference, sample, seed, strata=None):
    from vpp2.compact_kv.model import prepare_batch
    native=sample.get('_native_prepared')
    if native is not None:
        native={k:(v.to(reference.device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in native.items()}
    parts=list(prepare_batch(reference,sample,seed,prepared_inputs=native))
    runtime_cache=getattr(reference,'_full_mask_conditioning_cache',None)
    if native is None and runtime_cache is not None:
        keys=sample['_conditioning_key'].tolist()
        if len(keys)!=1:raise ValueError('On-demand cache currently requires verified microbatch one')
        runtime_cache.save(keys[0],parts[0])
    if strata is None:return tuple(parts)
    inputs=parts[0];action=inputs['action'];scheduler=reference.train_action_scheduler
    noises=[];times=[]
    for p,s in zip(sample['pilot_position'].tolist(),strata):
        gen=torch.Generator(device=action.device).manual_seed(seed_for(seed,'full-mask-action-noise',p))
        noises.append(torch.randn(action.shape[1:],device=action.device,dtype=action.dtype,generator=gen))
        u=(s+torch.rand((),device=action.device,generator=gen))/4
        times.append((scheduler._phi(u,scheduler.shift)*scheduler.num_train_timesteps).to(action.dtype))
    eps=torch.stack(noises);t=torch.stack(times)
    parts[3]=scheduler.add_noise(action,eps,t);parts[4]=t;parts[5]=scheduler.training_target(action,eps,t)
    return tuple(parts)
