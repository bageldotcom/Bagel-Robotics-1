"""Identities and deterministic training-only calibration panels."""
import json
from pathlib import Path
import numpy as np
import torch
from hydra.utils import instantiate
from vpp2.compact_kv.model import native_config, prepare_batch
from vpp2.compact_kv.spec import file_sha, stable_seed, atomic_json


def implementation_hashes():
    repo = Path(__file__).resolve().parents[3]
    return {str(p.relative_to(repo)): file_sha(p) for p in sorted((repo/'src/vpp2').rglob('*.py'))}


def training_dataset(config):
    return instantiate(native_config(config).data.train,
        metadata_path=str(Path(config['root'])/'data/prepared/train.csv'),
        is_training_set=False, val_set_proportion=0., balance_column=None, balance_weight_overrides=None)


def panel_entries(metadata, lengths, horizon, seed):
    """48 scoring + 16 diagnostics windows/task, distinct episodes; no validation data."""
    entries = []
    prefix = np.r_[0, np.cumsum(lengths)[:-1]]
    for task_id, task in enumerate(sorted(metadata.task_name.unique())):
        trajectories = np.flatnonzero(metadata.task_name.to_numpy() == task)
        if len(trajectories) < 64:
            raise ValueError(f'{task}: requires 64 distinct training episodes')
        gen = np.random.default_rng(stable_seed(seed, 'width-calibration', str(task)))
        for order, trajectory in enumerate(gen.permutation(trajectories)[:64]):
            last = max(0, int(lengths[trajectory]) - int(horizon))
            step = round((order % 3) * last / 2)
            row = metadata.iloc[trajectory]
            entries.append(dict(id=len(entries), task=str(task), task_id=task_id,
                index=int(prefix[trajectory])+step, trajectory=int(trajectory), step=step,
                episode_index=int(row.episode_index), partition='score' if order < 48 else 'diagnostic',
                half=order % 2, stratum=(order // 2) % 4))
    return entries


def prepare_calibration(reference, sample, entry, draw, config):
    """Two distinct video/action noise draws; action timesteps cover four strata."""
    position = -1_000_000 - (entry['id'] * 2 + draw)
    sample['pilot_position'] = torch.tensor([position])
    inputs, noise, tv, _, _, _, mask = prepare_batch(reference, sample, config['seed'])
    action = inputs['action']
    gen = torch.Generator(device=action.device).manual_seed(stable_seed(config['seed'], 'width-action', position))
    eps = torch.randn(action.shape, device=action.device, dtype=action.dtype, generator=gen)
    u = (entry['stratum'] + torch.rand(1, device=action.device, generator=gen)) / 4
    scheduler = reference.train_action_scheduler
    ta = (scheduler._phi(u, scheduler.shift) * scheduler.num_train_timesteps).to(action.dtype)
    noisy = scheduler.add_noise(action, eps, ta)
    target = scheduler.training_target(action, eps, ta)
    return inputs, noise, tv, noisy, ta, target, mask


def load_panel(config):
    import pandas as pd
    dataset = training_dataset(config)
    entries = panel_entries(dataset.metadata, dataset.trajectory_lengths, dataset.action_horizon, config['seed'])
    val = pd.read_csv(Path(config['root'])/'data/prepared/val.csv')
    if set(zip(dataset.metadata.task_name, dataset.metadata.episode_index)) & set(zip(val.task_name, val.episode_index)):
        raise RuntimeError('Training/validation episodes overlap')
    path = Path(config['root'])/'calibration/panel.json'
    manifest = dict(schema='width-training-panel-v1', entries=entries,
        train_sha256=file_sha(Path(config['root'])/'data/prepared/train.csv'),
        val_sha256=file_sha(Path(config['root'])/'data/prepared/val.csv'), seed=config['seed'])
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError('Calibration panel identity changed')
    return dataset, manifest
