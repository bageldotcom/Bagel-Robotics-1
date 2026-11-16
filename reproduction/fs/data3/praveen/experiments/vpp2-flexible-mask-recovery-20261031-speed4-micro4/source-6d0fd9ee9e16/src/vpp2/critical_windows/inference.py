"""Use VPP2.infer_action itself, replacing only the video prefill for masks."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from types import MethodType
import numpy as np
import torch
from vpp2.full_mask.panel import digest, file_hash, immutable
from vpp2.full_mask.curation import history_indices
from .contracts import read_json, checked_identity
from .metrics import native_queries


class NativeInferenceReader:
    """Observations are separate from scoring targets, including short tail chunks."""
    def __init__(self, config, manifest):
        from vpp2.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer
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


def deterministic_setup(local_rank):
    torch.cuda.set_device(local_rank)
    free,total=torch.cuda.mem_get_info()
    if free < 55 * 2**30:
        raise RuntimeError(f'Allocated GPU has only {free/2**30:.1f} GiB free; need 55 GiB before loading the native policy. Existing GPU work is left untouched.')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)


def source_identity():
    source=Path(__file__).resolve().parents[1]
    files=list((source/'models').rglob('*.py'))
    files += [Path(__file__),source/'learned_width_kv/masking.py',source/'full_mask/controller.py',
              source/'compact_kv/model.py',source/'checkpoint_compat.py',
              source/'datasets/lerobot/utils/normalizer.py',source/'critical_windows/contracts.py',
              source/'full_mask/curation.py',source/'full_mask/panel.py']
    return digest({str(p.relative_to(source)):file_hash(p) for p in sorted(set(files))})


def model_identity(config):
    source=read_json(Path(config['demonstrations'])/'manifest.json')
    expected=source['text_encoding']['teacher_identity']
    rows={}
    for name,key in [('video.pt','video_sha256'),('action.pt','action_sha256')]:
        path=Path(config['teacher_bundle'])/name
        actual=file_hash(path)
        if actual!=expected[key]:raise ValueError(f'Wrong teacher checkpoint: {path}')
        rows[name]=actual
    return rows


def smoke(config, manifest):
    from vpp2.compact_kv.model import load_reference
    from vpp2.learned_width_kv.masking import install_masks
    from vpp2.full_mask.runtime import model_versions, check_frozen
    deterministic_setup(0); model=load_reference(config);reader=NativeInferenceReader(config,manifest)
    weights=model_identity(config); versions=model_versions(model)
    chosen=[next(e for e in manifest['episodes'] if e['suite']==suite and e['role']=='fit') for suite in ('memory','precision')]
    saved=[]
    for e in chosen:
        start=native_queries(e['length'])[len(native_queries(e['length']))//2]
        inputs,_=reader.read(e['episode'],start)
        with torch.no_grad():
            first=model.infer_action(**inputs)['action'];second=model.infer_action(**inputs)['action']
        if not torch.equal(first,second):raise AssertionError('Native fixed-seed repeated sampling is not bitwise repeatable')
        saved.append((e,start,inputs,first))
    branch=install_masks(model.video_expert).eval();model.video_expert.use_gradient_checkpointing=False
    checks=[]
    for e,start,inputs,native in saved:
        with torch.no_grad(),masked_prefill(model,branch):
            opened=model.infer_action(**inputs)['action']
        if not torch.equal(native,opened):
            raise AssertionError(f'All-open sampled action parity failed: max_abs={float((native-opened).abs().max())}')
        checks.append(dict(episode=e['episode'],task=e['task'],frame=start,native_repeat_bitwise=True,all_open_bitwise=True))
    check_frozen(model,versions)
    receipt=dict(passed=True,manifest_identity=manifest['identity'],weights=weights,source_identity=source_identity(),checks=checks,
                 native_sampler_used=True,steps=10,horizon=32,replan=24,ground_truth_not_input=True,
                 policy_retention_established=False,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
    immutable(Path(config['root'])/'checks/native_inference_smoke.json',receipt)
    print(json.dumps(receipt),flush=True)


def run(config, manifest, role='monitor', mask_checkpoint=None):
    from vpp2.compact_kv.model import load_reference
    from vpp2.learned_width_kv.masking import install_masks
    from vpp2.full_mask.controller import Controller
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    deterministic_setup(local)
    receipt=read_json(Path(config['root'])/'checks/native_inference_smoke.json')
    if receipt.get('passed') is not True or receipt['manifest_identity']!=manifest['identity']:
        raise ValueError('Run the matching native-inference smoke check first')
    if receipt.get('source_identity')!=source_identity():raise ValueError('Inference source changed since smoke')
    weights=model_identity(config)
    if weights!=receipt['weights']:raise ValueError('Checkpoint changed since smoke')
    variant='native';extra={};controller=None
    if mask_checkpoint:
        state=torch.load(mask_checkpoint,map_location='cpu',weights_only=False)
        if 'controller' not in state or 'update' not in state:
            raise ValueError('Require a full_mask controller checkpoint; do not guess old mask formats')
        extra=dict(mask_sha256=file_hash(mask_checkpoint),update=int(state['update']))
        variant='hard_'+extra['mask_sha256'][:12]
        controller=Controller(config);controller.load_state_dict(state['controller'],strict=True)
    identity=digest(dict(manifest=manifest['identity'],weights=weights,settings=config['native_inference'],variant=variant,extra=extra,role=role,source=source_identity()))
    out=Path(config['root'])/'predictions'/identity[:16];out.mkdir(parents=True,exist_ok=True)
    if rank==0:immutable(out/'identity.json',dict(identity=identity,manifest_identity=manifest['identity'],variant=variant,weights=weights,extra=extra,role=role,settings=config['native_inference'],source_identity=source_identity()))
    model=load_reference(config);reader=NativeInferenceReader(config,manifest)
    branch=None;gates=None
    if controller is not None:
        branch=install_masks(model.video_expert).eval();controller=controller.cuda()
        gates=controller.gates(extra['update'],'binary')
    episodes=[e for e in manifest['episodes'] if e['role']==role]
    done=0;started=time.monotonic()
    for index,e in enumerate(episodes):
        if index%world!=rank:continue
        for start in native_queries(e['length']):
            path=out/f"{e['episode']:05d}_{start:05d}.npz"
            if path.exists():
                with np.load(path,allow_pickle=False) as old:
                    if str(old['identity'])!=identity or int(old['episode'])!=e['episode'] or int(old['frame'])!=start:
                        raise ValueError('Stale/incorrect prediction shard')
                done+=1;continue
            inputs,target=reader.read(e['episode'],start)
            with torch.no_grad():
                if branch is None:pred=model.infer_action(**inputs)['action']
                else:
                    with masked_prefill(model,branch,*gates):pred=model.infer_action(**inputs)['action']
            if pred.shape!=(32,16) or not torch.isfinite(pred).all():raise FloatingPointError('Invalid final sampled actions')
            physical=reader.normalizers['action'].backward(pred).numpy()
            tmp=path.with_suffix('.partial')
            with tmp.open('wb') as f:
                np.savez_compressed(f,identity=np.array(identity),episode=e['episode'],frame=start,
                    task=np.array(e['task']),suite=np.array(e['suite']),variant=np.array(variant),
                    normalized=pred.numpy(),physical=physical,target_normalized=target['normalized'],
                    target_physical=target['physical'],valid_steps=target['valid_steps'])
            tmp.replace(path);done+=1
            if done%20==0:print(json.dumps(dict(event='native_action_queries',rank=rank,complete=done,seconds=time.monotonic()-started,variant=variant)),flush=True)
    immutable(out/f'rank_{rank:02d}_complete.json',dict(identity=identity,rank=rank,world=world,queries=done,seconds=time.monotonic()-started))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--role',choices=['fit','monitor'],default='monitor');p.add_argument('--mask-checkpoint')
    a=p.parse_args();c=read_json(a.config);m=checked_identity(read_json(Path(c['root'])/'audit_manifest.json'))
    if a.smoke:smoke(c,m)
    else:run(c,m,a.role,a.mask_checkpoint)


if __name__=='__main__':main()
