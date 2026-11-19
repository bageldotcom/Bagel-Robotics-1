"""Episode-disjoint native action monitoring with isolated policy inputs."""

from collections import defaultdict

from contextlib import contextmanager

import json

from pathlib import Path

from types import MethodType

import numpy as np

import torch

from bagel_robotics.data_selection.manifest import file_hash,seed_for

from bagel_robotics.data_selection.descriptors import history_indices

def read_json(path):
    return json.loads(Path(path).read_text())

def select_windows(panel,config):
    by_task=defaultdict(lambda:defaultdict(list))
    for r in panel['windows']:
        if r['role']=='monitor':by_task[r['task']][r['episode']].append(r)
    selected=[];count=config['train']['native_windows_per_task']
    for task in panel['tasks']:
        groups=by_task[task]
        if len(groups)<count:raise ValueError('Not enough independent native-monitor episodes')
        # Verified critical monitoring windows are kept; unknown phases stay unknown.
        critical=[r for rs in groups.values() for r in rs if (r.get('critical_steps') or 0)>0]
        chosen={}
        for r in sorted(critical,key=lambda r:(r['episode'],r['frame'])):
            if len(chosen)<count:chosen.setdefault(r['episode'],r)
        episodes=sorted(groups,key=lambda ep:seed_for(config['seed'],'native-monitor-episode',task,ep))
        for ep in episodes:
            if len(chosen)>=count:break
            if ep in chosen:continue
            rows=sorted(groups[ep],key=lambda r:r['frame'])
            index=seed_for(config['seed'],'native-monitor-window',task,ep)%len(rows)
            chosen[ep]=rows[index]
        selected.extend(sorted(chosen.values(),key=lambda r:r['episode']))
    return selected

def reader_manifest(rows):
    episodes=[]
    for r in rows:
        episodes.append(dict(episode=r['episode'],task=r['task'],suite=r['suite'],role='monitor',
            length=r['length'],path=r['episode_path'],files=r['source_files'],text=r['text'],
            text_sha256=r['text_sha256'],prompt=r['prompt']))
    return dict(episodes=episodes)

class NativeInferenceReader:
    """Observations are separate from scoring targets, including short tail chunks."""
    def __init__(self, config, manifest):
        from bagel_robotics._internal.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer
        self.config = config; self.manifest = manifest; self.root = Path(config['demonstrations'])
        self.allowed = {e['episode']: e for e in manifest['episodes']}; self.current = None
        stats = read_json(self.root / 'dataset_stats.json')
        if stats != read_json(Path(config['teacher_bundle']) / 'dataset_stats.json'):
            raise ValueError('Native checkpoint normalization mismatch')
        self.normalizers = {field: SingleFieldLinearNormalizer(
            {k.removeprefix('global_'): torch.tensor(v, dtype=torch.float32)
             for k, v in stats[field]['default'].items() if k.startswith('global_')}, mode='z-score')
            for field in ('action', 'state')}
        self.text = {}

    def read(self, episode, start):
        e = self.allowed.get(episode)
        if e is None or e['role'] not in ('fit', 'monitor'):
            raise ValueError('Forbidden or sealed inference episode')
        if not 0 <= start < e['length']:
            raise ValueError('Query outside episode')
        if self.current != episode:
            base = self.root / e['path']
            self.arrays = {}
            for name in ('frames', 'actions', 'states'):
                path = base / (name + '.npy')
                if file_hash(path) != e['files'][name + '.npy']:
                    raise ValueError('Source changed after panel selection')
                self.arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
            if self.arrays['frames'].shape != (e['length'],3,240,416):
                raise ValueError('Native camera layout changed')
            self.current = episode
        if e['text'] not in self.text:
            path = self.root / e['text']
            if file_hash(path) != e['text_sha256']:
                raise ValueError('Text context changed')
            value = torch.load(path, map_location='cpu', weights_only=True)
            if value['prompt'] != e['prompt'] or value['context'].shape != (128,4096) or not value['mask'].all():
                raise ValueError('Native instruction context mismatch')
            self.text[e['text']] = value
        text = self.text[e['text']]
        # This is the ONLY image indexing in the policy-input path.
        ids = history_indices(start)
        history = torch.from_numpy(np.array(self.arrays['frames'][ids], copy=True)).float().div(255).mul(2).sub(1).permute(1,0,2,3)
        state = torch.from_numpy(np.array(self.arrays['states'][start],copy=True)).float()
        settings = self.config['native_inference']
        if (settings['steps'],settings['horizon'],settings['replan'],settings['history_stride'],settings['shift']) != (10,32,24,25,1.0):
            raise ValueError('Native inference contract changed')
        inputs = dict(prompt=None, input_image=history[:,0], condition_video=history,
                      proprio=self.normalizers['state'].forward(state), context=text['context'],
                      context_mask=text['mask'].bool(), action_horizon=32, num_video_frames=17,
                      num_inference_steps=10, sigma_shift=1.0, seed=settings['seed'],
                      video_seed=settings['seed'], rand_device=settings['rand_device'], tiled=False)
        # Targets are never arguments to infer_action. Padded entries are excluded.
        valid = min(32, e['length'] - start)
        target = np.zeros((32,16), dtype=np.float32)
        target[:valid] = self.arrays['actions'][start:start + valid]
        target[valid:] = target[valid-1]
        normalized = self.normalizers['action'].forward(torch.from_numpy(target)).numpy()
        return inputs, dict(physical=target, normalized=normalized, valid_steps=valid,
                            history_indices=ids)

@contextmanager
def masked_prefill(model, branch, hidden=None, ffn=None):
    """Single-worker scoped override; the native action sampler is untouched."""
    previous = model._build_frozen_video_cache
    had_override = '_build_frozen_video_cache' in model.__dict__
    original_override = model.__dict__.get('_build_frozen_video_cache')
    def replacement(self, *, latents, timestep_video, context, context_mask,
                    condition_latents, clip_fea, action_seq_len,
                    fuse_vae_embedding_in_latents=False, return_video_prediction=False,
                    max_layers=None, max_video_layers=None, action_visible_video_frames=None,
                    allow_trainable_video=False):
        if (return_video_prediction or max_layers is not None or max_video_layers is not None or
                fuse_vae_embedding_in_latents or action_visible_video_frames is not None or allow_trainable_video):
            raise ValueError('Unexpected non-native inference cache route')
        if tuple(latents.shape) != (1,16,5,30,52) or action_seq_len != 32:
            raise ValueError('Native video/action dimensions changed')
        mask = self._build_mot_attention_mask(1950,32,390,latents.device)
        inputs = dict(context=context, context_mask=context_mask,
                      condition_latents=condition_latents, clip_fea=clip_fea)
        caches = branch(inputs,latents,timestep_video,mask[:1950,:1950],hidden,ffn)
        return caches,mask,1950
    model._build_frozen_video_cache = MethodType(replacement,model)
    try:
        yield
    finally:
        if had_override:
            model._build_frozen_video_cache = original_override
        else:
            delattr(model,'_build_frozen_video_cache')
