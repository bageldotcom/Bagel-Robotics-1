"""Fixed, task-stratified native held-out windows; no training-set selection."""
from pathlib import Path
import time

import torch
import torch.distributed as dist
from hydra.utils import instantiate
from torch.utils.data import default_collate

from vpp2.evaluation import HeldoutActionEvaluator
from .model import native_config, prepare_batch, native_cache, action_prediction, infer_from_cache
from .losses import action_error, kv_terms, kv_plane_terms
from .spec import atomic_json


def make_panel(config):
    native = native_config(config)
    dataset = instantiate(native.data.train,
        metadata_path=str(Path(config['root']) / 'data/prepared/val.csv'),
        is_training_set=False, val_set_proportion=0., balance_column=None, balance_weight_overrides=None)
    evaluator = HeldoutActionEvaluator(dataset, windows_per_episode=16, num_inference_steps=10, seed=config['seed'])
    if len(evaluator.items) != 128 or len(evaluator.tasks) != 8:
        raise RuntimeError('Expected 128 fixed held-out windows across eight Precision tasks')
    return evaluator


@torch.no_grad()
def evaluate(engine, reference, panel, config, update, directory, limit=None):
    rank, world = dist.get_rank(), dist.get_world_size()
    items = list(range(len(panel.items)))
    if limit is not None:
        # First window of each task for the distributed smoke, eight total.
        items = [i for i, item in enumerate(panel.items) if i == 0 or item[1] != panel.items[i-1][1]]
    if len(items) % world:
        raise ValueError('ZeRO validation requires the same number of forwards on every rank')
    engine.eval()
    reference.mot.training = False
    local = []
    start = time.monotonic()
    for item_id in items[rank::world]:
        index, task_id, trajectory, step = panel.items[item_id]
        sample = panel.dataset._get(index)
        if int(sample['trajectory_idx']) != trajectory or int(sample['step_idx']) != step:
            raise RuntimeError('Fixed held-out sample identity changed')
        # Negative IDs separate validation randomness from training positions.
        position = -(item_id + 1)
        sample['pilot_position'] = torch.tensor(position)
        prepared = prepare_batch(reference, default_collate([sample]), config['seed'])
        inputs, noise, t_v, noisy, t_a, target, mask = prepared
        teacher = native_cache(reference, inputs, noise, t_v, mask)
        student = engine(inputs, noise, t_v, mask[:1950, :1950])
        if len(student) != 40 or len(teacher) != 40:
            raise RuntimeError('All 40 original K/V interfaces are required')
        t_pred = action_prediction(reference, inputs, teacher, noisy, t_a, mask)
        s_pred = action_prediction(reference, inputs, student, noisy, t_a, mask)
        weights = reference.train_action_scheduler.training_weight(t_a)
        train_mask = inputs.get('action_loss_is_pad', inputs.get('action_is_pad'))
        terms = kv_terms(student, teacher)
        teacher_chunk = infer_from_cache(reference, inputs, teacher, mask, [position], config['seed'])
        student_chunk = infer_from_cache(reference, inputs, student, mask, [position], config['seed'])
        # Offline chunk metrics use real action targets only, per native evaluator.
        real_mask = inputs.get('action_is_pad')
        if rank == 0 and item_id == 0:
            def cpu(value):
                if isinstance(value, torch.Tensor):
                    return value.detach().cpu()
                if isinstance(value, dict):
                    return {k: cpu(v) for k, v in value.items()}
                return value
            fixture = dict(inputs=cpu(inputs), noise=cpu(noise), t_v=cpu(t_v),
                           noisy_action=cpu(noisy), t_a=cpu(t_a), mask=cpu(mask), position=position,
                           expected_flow=cpu(s_pred), expected_chunk=cpu(student_chunk), update=update)
            folder = Path(directory) / 'validation'
            folder.mkdir(parents=True, exist_ok=True)
            torch.save(fixture, folder / f'fixture_{update:06d}.pt')
        row = dict(item_id=item_id, task=panel.tasks[task_id], trajectory=trajectory, step=step,
            teacher_flow_gt=float((action_error(t_pred, target, train_mask) * weights).mean()),
            student_flow_gt=float((action_error(s_pred, target, train_mask) * weights).mean()),
            action_teacher=float((action_error(s_pred, t_pred, train_mask) * weights).mean()),
            teacher_chunk_mse=float(action_error(teacher_chunk, inputs['action'], real_mask).mean()),
            student_chunk_mse=float(action_error(student_chunk, inputs['action'], real_mask).mean()),
            chunk_teacher_mse=float(action_error(student_chunk, teacher_chunk, real_mask).mean()),
            kv_raw=float(terms.mean()), kv_by_layer_kv=terms.mean(0).cpu().tolist())
        if config.get("width_temporal_diagnostics", False):
            row["kv_by_plane_layer_kv"] = kv_plane_terms(student, teacher).mean(0).cpu().tolist()
        local.append(row)
        del student, teacher, prepared, inputs, teacher_chunk, student_chunk
    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    result = None
    if rank == 0:
        rows = sorted([x for shard in gathered for x in shard], key=lambda x: x['item_id'])
        if len({x['item_id'] for x in rows}) != len(items):
            raise RuntimeError('Duplicated/missing validation windows')
        metric_keys = ['teacher_flow_gt', 'student_flow_gt', 'action_teacher', 'teacher_chunk_mse',
                       'student_chunk_mse', 'chunk_teacher_mse', 'kv_raw']
        by_task = {}
        for task in panel.tasks:
            group = [x for x in rows if x['task'] == task]
            by_task[task] = {key: sum(x[key] for x in group)/len(group) for key in metric_keys}
        result = dict(kind='validation', update=update, windows=len(rows), by_task=by_task,
                      seconds=time.monotonic()-start)
        result.update({key: sum(x[key] for x in rows)/len(rows) for key in metric_keys})
        result['kv_by_layer_kv'] = torch.tensor([x['kv_by_layer_kv'] for x in rows]).mean(0).tolist()
        if config.get('width_temporal_diagnostics', False):
            result['kv_by_plane_layer_kv'] = torch.tensor([x['kv_by_plane_layer_kv'] for x in rows]).mean(0).tolist()
        atomic_json(Path(directory) / f'validation/update_{update:06d}.json', dict(summary=result, windows=rows))
    dist.barrier()
    engine.train()
    reference.mot.training = True
    return result
