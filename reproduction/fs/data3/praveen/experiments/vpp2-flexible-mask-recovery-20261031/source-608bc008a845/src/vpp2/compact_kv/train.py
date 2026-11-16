"""Precision-only compact-video training with frozen native Action2B."""
import argparse
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, default_collate
from hydra.utils import instantiate
import deepspeed

from .blocks import install_blocks
from .checkpointing import LosslessCheckpointEngine, save_training, export_native, restore_rng
from .gpu_check import source_hashes, tensor_hash
from .losses import action_error, kv_terms
from .model import native_config, load_student, load_reference, prepare_batch, native_cache, action_prediction, VideoBranch
from .reporting import Reporter
from .sampling import PositionDataset, GlobalBatchSampler
from .spec import CANDIDATES, architecture, atomic_json, file_sha, lr_multiplier, kv_weight, stable_seed
from .validation import make_panel, evaluate


STOP = False

def stop_requested(*_):
    global STOP
    STOP = True


def ds_config(config, world):
    micro = config['micro_batch']
    if config['global_batch'] % (world * micro):
        raise ValueError('Global batch must divide exactly')
    zero = dict(stage=3, overlap_comm=True, contiguous_gradients=True,
                reduce_bucket_size=20_000_000, stage3_prefetch_bucket_size=20_000_000,
                stage3_param_persistence_threshold=100_000,
                stage3_max_live_parameters=600_000_000,
                stage3_max_reuse_distance=600_000_000,
                stage3_gather_16bit_weights_on_model_save=True)
    if config['optimizer_cpu_offload']:
        zero['offload_optimizer'] = dict(device='cpu', pin_memory=True)
    return dict(train_batch_size=config['global_batch'], train_micro_batch_size_per_gpu=micro,
                gradient_accumulation_steps=config['global_batch'] // (world * micro),
                gradient_clipping=config['gradient_clipping'],
                bf16=dict(enabled=True), zero_optimization=zero,
                zero_allow_untested_optimizer=True, communication_data_type='fp32',
                steps_per_print=1000000, wall_clock_breakdown=False)


def assert_student_geometry(student, teacher):
    if len(student) != 40 or len(teacher) != 40:
        raise RuntimeError('Every original layer must expose K and V')
    for i, (a, b) in enumerate(zip(student, teacher)):
        for key in ('k', 'v'):
            if tuple(a[key].shape) != tuple(b[key].shape) or tuple(a[key].shape[1:]) != (1950, 5120):
                raise RuntimeError(f'Wrong {key} interface at original layer {i}')
            if b[key].requires_grad:
                raise RuntimeError('Teacher targets must be detached')



def authorize_fast_input_resume(old, new, receipt_path):
    """Permit only the audited no-target-VAE optimization; retain experiment identity."""
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt.get('schema') != 'compact-fast-inputs-v1' or not receipt.get('passed'):
        raise RuntimeError('Invalid fast-input equivalence receipt')
    for key in ('teacher', 'data_ready_sha256', 'architecture'):
        if old[key] != new[key]:
            raise RuntimeError('Resume changes scientific identity: ' + key)
    if receipt['old_implementation_hashes'] != old['implementation_hashes'] or receipt['new_implementation_hashes'] != new['implementation_hashes']:
        raise RuntimeError('Equivalence proof does not cover these exact source versions')
    changed = {k for k in set(old['implementation_hashes']) | set(new['implementation_hashes'])
               if old['implementation_hashes'].get(k) != new['implementation_hashes'].get(k)}
    allowed = {'src/vpp2/compact_kv/model.py', 'src/vpp2/compact_kv/train.py',
               'src/vpp2/models/wan21_14b/vpp2.py'}
    if changed != allowed or set(receipt['changed_files']) != changed:
        raise RuntimeError('Unexpected implementation change during optimization resume')
    for key in ('prepared_inputs_exact', 'all_40_kv_exact', 'action_flow_exact', 'rng_states_exact'):
        if not receipt['checks'].get(key):
            raise RuntimeError('Missing equivalence check: ' + key)
    for key in ('action_sha256', 'proprio_sha256'):
        if old['native_gpu_checks'][key] != new['native_gpu_checks'][key]:
            raise RuntimeError('Frozen weights changed during optimization')
    return receipt


def assert_tree_exact(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        if set(left) != set(right): raise RuntimeError('Prepared input keys differ')
        for key in left: assert_tree_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right): raise RuntimeError('Prepared input lengths differ')
        for a, b in zip(left, right): assert_tree_exact(a, b)
    elif left != right:
        raise RuntimeError('Prepared input metadata differs')


def authorize_pilot_resume(old, new, receipt_path):
    """Allow only a verified stopping-point change, with the training loop intact."""
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt.get('schema') != 'compact-pilot-duration-v1' or not receipt.get('passed'):
        raise RuntimeError('Invalid pilot-duration migration receipt')
    for key in ('teacher', 'data_ready_sha256', 'architecture', 'native_gpu_checks', 'source_hashes'):
        if old[key] != new[key]:
            raise RuntimeError('Pilot duration changes scientific identity: ' + key)
    if receipt['old_implementation_hashes'] != old['implementation_hashes'] or receipt['new_implementation_hashes'] != new['implementation_hashes']:
        raise RuntimeError('Pilot-duration proof does not cover these source versions')
    changed = {k for k in set(old['implementation_hashes']) | set(new['implementation_hashes'])
               if old['implementation_hashes'].get(k) != new['implementation_hashes'].get(k)}
    if changed != {'src/vpp2/compact_kv/train.py'} or set(receipt['changed_files']) != changed:
        raise RuntimeError('Only training orchestration may change for a shorter pilot')
    for key in ('training_loop_ast_exact', 'optimizer_setup_ast_exact', 'config_unchanged',
                'schedule_1000_retained', 'sampling_noise_prefix_exact', 'cpu_tests_passed'):
        if not receipt['checks'].get(key):
            raise RuntimeError('Missing pilot-duration check: ' + key)
    return receipt


def pilot_end(config, start, smoke_updates=0, stop_after_update=None):
    """The comparison endpoint is independent of the original LR schedule horizon."""
    if stop_after_update is not None and stop_after_update not in (500, 1000):
        raise ValueError('Supported main pilot endpoints are 500 and 1000')
    end = start + smoke_updates if smoke_updates else (stop_after_update or config['max_updates'])
    if not 0 <= start < end <= config['max_updates']:
        raise ValueError('Pilot endpoint must follow the restored update and fit the LR horizon')
    return end


def train(args):
    config = json.loads(Path(args.config).read_text())
    if args.micro_batch is not None:
        config['micro_batch'] = args.micro_batch
    if args.shard_teacher:
        config['shard_frozen_teacher'] = True
    if args.offload_optimizer:
        config['optimizer_cpu_offload'] = True
    if config['max_updates'] != 1000 or config['global_batch'] != 288:
        raise ValueError('Pilot scientific contract requires 1000 updates, global batch 288')
    execution_policy = None
    if args.candidate in ('W1', 'W2'):
        from vpp2.width_kv.reproducibility import configure_deterministic_training
        execution_policy = configure_deterministic_training()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=40))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8:
        raise ValueError('These pilots require eight allocated GPUs')
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, stop_requested)
    root = Path(config['root'])
    receipt = json.loads((root / 'checks/native_gpu.json').read_text())
    if not receipt['passed'] or receipt['source_hashes'] != source_hashes():
        raise RuntimeError('GPU correctness receipt does not match the implementation')
    repo = Path(__file__).resolve().parents[3]
    implementation_hashes = {str(p.relative_to(repo)): file_sha(p)
                            for p in sorted((repo / 'src/vpp2').rglob('*.py'))}
    teacher_identity = json.loads((root / 'teacher_identity.json').read_text())
    selection = None
    if args.candidate in ('W1', 'W2'):
        selection_path = root / 'calibration' / f'{args.candidate}_selection.json'
        selection = json.loads(selection_path.read_text())
        calibration = json.loads((root / 'calibration/COMPLETE.json').read_text())
        if not calibration['passed'] or calibration['configuration_sha256'] != file_sha(args.config):
            raise RuntimeError('Calibration configuration changed or calibration failed')
        if calibration['selections'][args.candidate] != file_sha(selection_path):
            raise RuntimeError('Channel selection changed after calibration')
        if calibration['implementation_hashes'] != implementation_hashes or calibration['teacher'] != teacher_identity:
            raise RuntimeError('Calibration belongs to different source or teacher')
        if not args.smoke_updates:
            integration = json.loads((root / 'checks/width_integration.json').read_text())
            if integration['micro_batch'] != config['micro_batch'] or integration['global_batch'] != config['global_batch']:
                raise RuntimeError('Main batch configuration differs from GPU integration checks')
            if not integration['passed'] or integration['implementation_hashes'] != implementation_hashes:
                raise RuntimeError('Eight-GPU width integration checks not passed for this code')
    provenance = dict(teacher=teacher_identity, data_ready_sha256=file_sha(root / 'data/ready.json'),
                      native_gpu_checks=receipt, architecture=architecture(args.candidate, selection), source_hashes=source_hashes(), implementation_hashes=implementation_hashes)
    if execution_policy is not None:
        provenance['execution_policy'] = execution_policy
    directory = root / (args.smoke_name if args.smoke_updates else 'runs') / args.candidate
    directory.mkdir(parents=True, exist_ok=True)
    if not args.resume and (directory / 'checkpoints/latest').exists():
        raise RuntimeError('Existing optimizer state: use explicit --resume or a fresh directory')
    if rank == 0:
        atomic_json(directory / 'config.json', config)
        atomic_json(directory / 'provenance.json', provenance)
        print(json.dumps(dict(event='initializing_student', candidate=args.candidate,
                              global_batch=288, micro=config['micro_batch'], world=world)), flush=True)
    student = load_student(config, args.candidate)
    if rank == 0 and hasattr(student, 'width_transfer_audit'):
        atomic_json(directory / 'width_transfer_audit.json', student.width_transfer_audit)
    params = [p for p in student.parameters() if p.requires_grad]
    if config['optimizer_cpu_offload']:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        optimizer = DeepSpeedCPUAdam(params, lr=config['learning_rate'], betas=tuple(config['betas']),
                                     weight_decay=config['weight_decay'], adamw_mode=True)
    else:
        optimizer = torch.optim.AdamW(params, lr=config['learning_rate'], betas=tuple(config['betas']),
                                      weight_decay=config['weight_decay'], fused=True)
    engine, _, _, _ = deepspeed.initialize(model=student, optimizer=optimizer,
                                           config=ds_config(config, world), dist_init_required=False)
    engine.checkpoint_engine = LosslessCheckpointEngine()
    start = 0
    resume_rng = None
    if args.resume:
        path, client = engine.load_checkpoint(str(directory / 'checkpoints'), tag=None,
                                              load_optimizer_states=True, load_lr_scheduler_states=False)
        if not path or client['config'] != config:
            raise RuntimeError('Checkpoint/config mismatch')
        if client['provenance'] != provenance:
            if not args.resume_equivalence_receipt:
                raise RuntimeError('Source changed without a verified behavior-equivalence receipt')
            migration = json.loads(Path(args.resume_equivalence_receipt).read_text())
            if migration.get('schema') == 'compact-pilot-duration-v1':
                authorize_pilot_resume(client['provenance'], provenance, args.resume_equivalence_receipt)
            else:
                authorize_fast_input_resume(client['provenance'], provenance, args.resume_equivalence_receipt)
            if rank == 0:
                atomic_json(directory / f'code_transition_at_update_{client["update"]:06d}.json',
                    dict(old_provenance=client['provenance'], new_provenance=provenance,
                         equivalence_receipt=str(Path(args.resume_equivalence_receipt).resolve()),
                         equivalence_sha256=file_sha(args.resume_equivalence_receipt),
                         optimizer_and_sample_sequence_preserved=True))
        start = int(client['update'])
        if client['next_global_sample'] != start * config['global_batch']:
            raise RuntimeError('Corrupt sampler cursor')
        resume_rng = client['rank_rng'][rank]
    if rank == 0:
        print(json.dumps(dict(event='loading_frozen_teacher', allocated_GiB=torch.cuda.memory_allocated()/2**30)), flush=True)
    reference = load_reference(config, device=f'cuda:{local_rank}')
    # Native action checkpointing is compatible with frozen action weights;
    # autograd must still traverse these operations into the student K/V tensors.
    reference.mot.training = True
    action_sha = tensor_hash(reference.action_expert) if rank == 0 else None
    if rank == 0 and action_sha != receipt['action_sha256']:
        raise RuntimeError('Frozen action checkpoint identity changed')
    if any(p.requires_grad for p in reference.parameters()):
        raise RuntimeError('All teacher/encoder/action/proprio parameters must be frozen')
    dataset = PositionDataset(instantiate(native_config(config).data.train), config['seed'])
    end = pilot_end(config, start, args.smoke_updates, args.stop_after_update)
    if rank == 0:
        atomic_json(directory / 'pilot_budget.json', dict(comparison_update=end,
            lr_schedule_updates=config['max_updates'], starting_update=start,
            optimizer_reset=not bool(args.resume), simulation_enabled=False))
    sampler = GlobalBatchSampler(start, end, rank, world, config['micro_batch'], config['global_batch'])
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=config['num_workers'], pin_memory=True,
                        persistent_workers=config['num_workers'] > 0,
                        generator=torch.Generator().manual_seed(stable_seed(config['seed'], 'loader', rank)),
                        **(dict(prefetch_factor=2, multiprocessing_context='spawn') if config['num_workers'] else {}))
    panel = make_panel(config)
    if config.get('shard_frozen_teacher', False):
        # Pin exact real-window teacher K/V before replacing replicated storage
        # with ZeRO-3 partitioning. This is a storage/execution change only.
        index, _, _, _ = panel.items[rank * 16]
        check_sample = panel.dataset._get(index)
        check_sample['pilot_position'] = torch.tensor(-(rank*16+1))
        check_inputs, check_noise, check_t, _, _, _, check_mask = prepare_batch(
            reference, default_collate([check_sample]), config['seed'])
        with torch.no_grad():
            before = native_cache(reference, check_inputs, check_noise, check_t, check_mask)
            before = [{k: v.cpu() for k, v in layer.items()} for layer in before]
        install_blocks(reference.video_expert, ())
        teacher_branch = VideoBranch(reference.video_expert).eval().requires_grad_(False)
        teacher_ds_config = ds_config(config, world)
        teacher_ds_config['zero_optimization'].pop('offload_optimizer', None)
        teacher_engine, _, _, _ = deepspeed.initialize(model=teacher_branch,
                config=teacher_ds_config, dist_init_required=False)
        teacher_engine.eval()
        object.__setattr__(reference, '_compact_teacher_engine', teacher_engine)
        with torch.no_grad():
            after = native_cache(reference, check_inputs, check_noise, check_t, check_mask)
            for layer, (a, b) in enumerate(zip(before, after)):
                for key in ('k', 'v'):
                    torch.testing.assert_close(a[key], b[key].cpu(), rtol=0, atol=0,
                            msg=f'Frozen teacher sharding changed layer {layer}/{key}')
        del before, after, check_inputs, check_noise, check_t, check_mask, check_sample
        gc.collect(); torch.cuda.empty_cache()
        dist.barrier()
        if rank == 0:
            atomic_json(directory / 'teacher_sharding_check.json', dict(passed=True,
                original_layer_slots=40, exact_k_and_v=True, windows=8))
            print('PASS: sharded frozen teacher matches all 40 K/V layers exactly on eight held-out windows', flush=True)
    reporter = Reporter(directory, config, args.candidate, enabled=not bool(args.smoke_updates)) if rank == 0 else None
    if rank == 0 and reporter.wandb is not None:
        reporter.wandb.name = f'{args.candidate}-precision-{end}'
        reporter.wandb.config.update(dict(pilot_comparison_updates=end,
            lr_schedule_updates=config['max_updates']), allow_val_change=True)
    if start > 0:
        fixture_path = directory / f'validation/fixture_{start:06d}.pt'
        engine.eval(); reference.mot.training = False
        if fixture_path.exists():
            fixture = torch.load(fixture_path, map_location='cpu', weights_only=True)
            def to_device(value):
                if isinstance(value, torch.Tensor): return value.to(f'cuda:{local_rank}')
                if isinstance(value, dict): return {k:to_device(v) for k,v in value.items()}
                return value
            inputs = to_device(fixture['inputs']); mask = to_device(fixture['mask'])
            with torch.no_grad():
                cache = engine(inputs, to_device(fixture['noise']), to_device(fixture['t_v']), mask[:1950,:1950])
                prediction = action_prediction(reference, inputs, cache, to_device(fixture['noisy_action']),
                                               to_device(fixture['t_a']), mask)
            torch.testing.assert_close(prediction.cpu(), fixture['expected_flow'], rtol=0, atol=0)
            del fixture, inputs, cache, prediction, mask
            check_type = 'saved_pre_pause_prediction'
        else:
            # Older checkpoints saved on SIGUSR1 did not save a validation
            # fixture off milestone. Check both preparation implementations on
            # the restored student before any optimizer update instead.
            if not args.resume_equivalence_receipt:
                raise RuntimeError('Resume fixture missing without explicit optimization migration')
            item_id = rank * 16
            index = panel.items[item_id][0]
            sample = panel.dataset._get(index)
            sample['pilot_position'] = torch.tensor(-(item_id+1))
            sample = default_collate([sample])
            legacy = prepare_batch(reference, sample, config['seed'], encode_video_target=True)
            fast = prepare_batch(reference, sample, config['seed'])
            assert_tree_exact(legacy, fast)
            predictions = []
            with torch.no_grad():
                for prepared in (legacy, fast):
                    inputs, noise, t_v, noisy, t_a, _, mask = prepared
                    cache = engine(inputs, noise, t_v, mask[:1950,:1950])
                    predictions.append(action_prediction(reference, inputs, cache, noisy, t_a, mask).cpu())
                    del cache
            torch.testing.assert_close(*predictions, rtol=0, atol=0)
            del legacy, fast, predictions, sample, prepared, inputs, noise, t_v, noisy, t_a, mask
            check_type = 'legacy_vs_fast_on_restored_student'
        if engine.global_steps != start:
            raise RuntimeError('DeepSpeed optimizer update counter was not restored')
        if rank == 0:
            atomic_json(directory / f'resume_check_{start:06d}.json', dict(passed=True, update=start,
                model_predictions_exact=True, check_type=check_type, next_global_sample=start*config['global_batch']))
    if start == 0:
        metrics = evaluate(engine, reference, panel, config, 0, directory,
                           limit=8 if args.smoke_updates else None)
        if rank == 0:
            reporter.write(metrics)
            print(json.dumps(dict(event='initial_validation', **metrics)), flush=True)
    engine.train()
    reference.mot.training = True
    accumulation = sampler.accumulation
    micro_index, update = 0, start
    stop = False
    profile_batches = []
    accum = torch.zeros(85, device=f'cuda:{local_rank}', dtype=torch.float64)
    if resume_rng is not None:
        restore_rng(resume_rng)
    step_start = time.monotonic()
    for sample in loader:
        lr = config['learning_rate'] * lr_multiplier(update, config['max_updates'],
                        config['warmup_updates'], config['lr_final_fraction'])
        for group in engine.optimizer.param_groups:
            group['lr'] = lr
        coefficient = kv_weight(update + 1, config['kv_max_weight'], config['kv_ramp_updates'])
        profile_update = update < start + 3 or (update + 1) % 50 == 0
        timing_events = []
        def mark_stage(name):
            if profile_update:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                timing_events.append((name, event))
        mark_stage('start')
        prepared = prepare_batch(reference, sample, config['seed'])
        mark_stage('prepare_inputs')
        inputs, noise, t_v, noisy_action, t_a, target, mask = prepared
        with torch.no_grad():
            teacher_cache = native_cache(reference, inputs, noise, t_v, mask)
            mark_stage('teacher_video')
            teacher_action = action_prediction(reference, inputs, teacher_cache, noisy_action, t_a, mask)
            mark_stage('teacher_action')
        student_cache = engine(inputs, noise, t_v, mask[:1950, :1950])
        mark_stage('student_video')
        assert_student_geometry(student_cache, teacher_cache)
        student_action = action_prediction(reference, inputs, student_cache, noisy_action, t_a, mask)
        mark_stage('student_action')
        action_mask = inputs.get('action_loss_is_pad', inputs.get('action_is_pad'))
        time_weight = reference.train_action_scheduler.training_weight(t_a)
        gt = (action_error(student_action, target, action_mask) * time_weight).mean()
        action_kd = (action_error(student_action, teacher_action, action_mask) * time_weight).mean()
        terms = kv_terms(student_cache, teacher_cache)
        # [B,40,2]: ALL original layers; same-index K/V, all tokens, all heads.
        kv = terms.mean()
        loss = config['action_gt_weight']*gt + config['action_teacher_weight']*action_kd + coefficient*kv
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        values = torch.cat([torch.stack([gt.detach(), action_kd.detach(), kv.detach(), coefficient*kv.detach(), loss.detach()]),
                            terms.detach().mean(0).reshape(-1)]).to(accum.dtype)
        accum += values
        mark_stage('loss_computation')
        engine.backward(loss)
        mark_stage('backward')
        if any(p.grad is not None for p in reference.parameters()):
            raise RuntimeError('Frozen action/teacher unexpectedly accumulated parameter gradients')
        engine.step()
        mark_stage('optimizer_step')
        if profile_update:
            profile_batches.append(timing_events)
        micro_index += 1
        del student_cache, teacher_cache, student_action, teacher_action, terms, loss, prepared, inputs
        if micro_index % accumulation:
            if rank == 0 and args.smoke_updates:
                print(f'smoke microbatch {micro_index}/{accumulation * args.smoke_updates}', flush=True)
            continue
        update += 1
        gradient_norm = engine.get_global_grad_norm()
        if gradient_norm is None or not np.isfinite(float(gradient_norm)):
            raise FloatingPointError('Nonfinite or missing global video gradient norm')
        if engine.global_steps != update:
            raise RuntimeError('Optimizer and sampler update counters diverged')
        dist.all_reduce(accum, op=dist.ReduceOp.SUM)
        accum /= world * accumulation
        torch.cuda.synchronize()
        elapsed = time.monotonic() - step_start
        component_times = {}
        if profile_batches:
            names = [name for name, _ in profile_batches[0][1:]]
            durations = [sum(events[j][1].elapsed_time(events[j+1][1]) for events in profile_batches)/1000
                         for j in range(len(names))]
            times = torch.tensor(durations, device=f'cuda:{local_rank}', dtype=torch.float64)
            dist.all_reduce(times, op=dist.ReduceOp.SUM)
            times /= world
            component_times = {f'timing_{name}_seconds': float(value) for name, value in zip(names, times.cpu().tolist())}
            profile_batches.clear()
        peak = torch.tensor([torch.cuda.max_memory_allocated()], device=f'cuda:{local_rank}', dtype=torch.float64)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        if rank == 0:
            array = accum.cpu().tolist()
            row = dict(kind='train', update=update, action_gt=array[0], action_teacher=array[1],
                    kv_raw=array[2], kv_weighted=array[3], loss=array[4], kv_weight=coefficient,
                    kv_by_layer_kv=np.array(array[5:]).reshape(40,2).tolist(), learning_rate=lr,
                    seconds_per_update=elapsed, windows_per_second=config['global_batch']/elapsed,
                    max_gpu_allocated_GiB=float(peak)/2**30, global_windows=update*config['global_batch'],
                    global_batch=config['global_batch'], micro_batch=config['micro_batch'],
                    video_gradient_norm=float(gradient_norm), **component_times)
            reporter.write(row)
            print(json.dumps({k:v for k,v in row.items() if k!='kv_by_layer_kv'}), flush=True)
        accum.zero_()
        stopped = torch.tensor([int(STOP)], device=f'cuda:{local_rank}')
        dist.all_reduce(stopped, op=dist.ReduceOp.MAX)
        stop = bool(stopped.item())
        if update in config['validation_updates'] or (args.smoke_updates and update == end) or stop:
            metric = evaluate(engine, reference, panel, config, update, directory,
                              limit=8 if args.smoke_updates else None)
            if rank == 0:
                reporter.write(metric)
                print(json.dumps(dict(event='validation', **metric)), flush=True)
        if update % config['checkpoint_every'] == 0 or update == end or stop:
            save_training(engine, directory, config, args.candidate, update, provenance)
        if stop:
            break
        step_start = time.monotonic()
    if rank == 0:
        if tensor_hash(reference.action_expert) != action_sha:
            raise RuntimeError('Action expert weights changed during training')
    if update == end and not stop:
        exported = export_native(engine, directory, config, args.candidate, update, provenance)
        if rank == 0:
            atomic_json(directory / 'TRAINING_COMPLETE.json', dict(update=update, export=str(exported),
                       candidate=args.candidate, smoke=bool(args.smoke_updates), all_40_kv_loss=True,
                       lr_schedule_updates=config['max_updates'], comparison_update=end))
    if rank == 0:
        reporter.close()
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--candidate', choices=list(CANDIDATES), required=True)
    parser.add_argument('--smoke-updates', type=int, default=0)
    parser.add_argument('--micro-batch', type=int)
    parser.add_argument('--offload-optimizer', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-update', type=int, choices=(500, 1000),
                        help='Main pilot endpoint; the original 1000-update LR schedule remains unchanged')
    parser.add_argument('--resume-equivalence-receipt')
    parser.add_argument('--shard-teacher', action='store_true')
    parser.add_argument('--smoke-name', default='smoke')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
