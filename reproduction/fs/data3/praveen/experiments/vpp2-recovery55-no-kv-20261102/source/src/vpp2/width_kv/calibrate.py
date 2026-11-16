"""Frozen-teacher action-sensitivity channel selection on allocated GPUs only."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.utils.data import default_collate
from vpp2.compact_kv.blocks import install_blocks
from vpp2.compact_kv.model import load_reference, VideoBranch, native_cache, action_prediction
from vpp2.compact_kv.losses import action_error
from vpp2.compact_kv.gpu_check import tensor_hash, source_hashes
from vpp2.compact_kv.spec import atomic_json, file_sha, stable_seed
from .common import load_panel, prepare_calibration, implementation_hashes
from .slicing import standard_selection
from .scoring import install_gates, gradients, masked_probe, selection_from_statistics, task_scores


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True); args = parser.parse_args()
    config = json.loads(Path(args.config).read_text()); root = Path(config['root']); folder = root/'calibration'
    rank = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=40)); world = dist.get_world_size()
    if world != 8: raise RuntimeError('Calibration requires eight allocated GPUs')
    torch.manual_seed(config['seed']); torch.backends.cuda.matmul.allow_tf32 = False
    receipt = json.loads((root/'checks/native_gpu.json').read_text())
    if not receipt['passed'] or receipt['source_hashes'] != source_hashes(): raise RuntimeError('Native GPU checks missing/stale')
    if (folder/'COMPLETE.json').exists(): raise RuntimeError('Completed calibration is immutable; use a new experiment root')
    folder.mkdir(parents=True, exist_ok=True)
    dataset, panel = load_panel(config)
    if rank == 0: atomic_json(folder/'panel.json', panel)
    reference = load_reference(config, device=f'cuda:{rank}')
    original_action_sha = tensor_hash(reference.action_expert) if rank == 0 else None
    entries = [e for e in panel['entries'] if e['partition'] == 'score']
    first = entries[rank]
    sample = default_collate([dataset._get(first['index'])])
    prepared = prepare_calibration(reference, sample, first, 0, config)
    inputs, noise, tv, noisy, ta, target, mask = prepared
    with torch.no_grad():
        original_cache = native_cache(reference, inputs, noise, tv, mask)
        original_prediction = action_prediction(reference, inputs, original_cache, noisy, ta, mask).cpu()
        original_cache = [{k: v.cpu() for k,v in layer.items()} for layer in original_cache]
    install_blocks(reference.video_expert, ())
    reference.video_expert.use_gradient_checkpointing = True
    branch = VideoBranch(reference.video_expert).train()
    residual, ffn, handles = install_gates(reference.video_expert)
    gates = list(residual) + list(ffn)
    with torch.no_grad():
        gated = branch(inputs, noise, tv, mask[:1950,:1950])
        for a,b in zip(gated,original_cache):
            for key in ('k','v'): torch.testing.assert_close(a[key].cpu(),b[key],rtol=0,atol=0)
        prediction = action_prediction(reference,inputs,gated,noisy,ta,mask)
        torch.testing.assert_close(prediction.cpu(),original_prediction,rtol=0,atol=0)
    del prepared, inputs, noise, tv, noisy, ta, target, mask, gated, original_cache, original_prediction, prediction, sample
    reference.mot.training = True
    shape = {'residual': (2,8,40,5120), 'ffn': (2,8,40,13824)}
    behaviour = {k: torch.zeros(v,device=f'cuda:{rank}',dtype=torch.float32) for k,v in shape.items()}
    demonstration = {k: torch.zeros_like(v) for k,v in behaviour.items()}
    counts = torch.zeros(2,8,device=f'cuda:{rank}')
    start = time.monotonic()
    for n, entry in enumerate(entries[rank::world]):
        sample = default_collate([dataset._get(entry['index'])])
        for draw in range(2):
            inputs, noise, tv, noisy, ta, target, mask = prepare_calibration(reference,sample,entry,draw,config)
            cache = branch(inputs,noise,tv,mask[:1950,:1950])
            prediction = action_prediction(reference,inputs,cache,noisy,ta,mask)
            action_mask = inputs.get('action_loss_is_pad',inputs.get('action_is_pad'))
            h,t = entry['half'],entry['task_id']
            for probe in range(2):
                objective = masked_probe(prediction,action_mask,stable_seed(config['seed'],'width-probe',entry['id'],draw,probe))
                grad = gradients(objective,gates,retain_graph=True)
                behaviour['residual'][h,t] += torch.stack(grad[:40]).square()/2
                behaviour['ffn'][h,t] += torch.stack(grad[40:]).square()/2
                del grad, objective
            objective = (action_error(prediction,target,action_mask)*reference.train_action_scheduler.training_weight(ta)).mean()
            grad = gradients(objective,gates)
            demonstration['residual'][h,t] += torch.stack(grad[:40]).abs()
            demonstration['ffn'][h,t] += torch.stack(grad[40:]).abs()
            counts[h,t] += 1
            if any(p.grad is not None or p.requires_grad for p in reference.parameters()):
                raise RuntimeError('Calibration must not train any native parameter')
            del grad, objective, prediction, cache, inputs, noise, tv, noisy, ta, target, mask
        if n % 4 == 0:
            print(json.dumps(dict(event='calibration',rank=rank,windows_done=n+1,local_windows=len(entries[rank::world]),seconds=time.monotonic()-start)),flush=True)
    for tensor in [counts,*behaviour.values(),*demonstration.values()]: dist.all_reduce(tensor)
    if not (counts == 48).all(): raise RuntimeError('Scoring must contain 24 windows x two draws per task/half')
    for group in behaviour:
        behaviour[group] /= counts[:,:,None,None]
        demonstration[group] /= counts[:,:,None,None]
    if rank == 0:
        action_sha = tensor_hash(reference.action_expert)
        if action_sha != original_action_sha: raise RuntimeError('Frozen action weights changed')
        aggregate_b = {k:v.mean(0).cpu() for k,v in behaviour.items()}
        aggregate_d = {k:v.mean(0).cpu() for k,v in demonstration.items()}
        w1 = dict(standard_selection(),candidate='W1')
        w2 = dict(selection_from_statistics(aggregate_b,aggregate_d),candidate='W2')
        halves = [selection_from_statistics({k:v[i].cpu() for k,v in behaviour.items()},
                    {k:v[i].cpu() for k,v in demonstration.items()}) for i in range(2)]
        def overlap(a,b): return len(set(a)&set(b))/len(set(a)|set(b))
        consistency = dict(hidden_jaccard=overlap(halves[0]['hidden_indices'],halves[1]['hidden_indices']),
            ffn_jaccard=[overlap(a,b) for a,b in zip(halves[0]['ffn_indices'],halves[1]['ffn_indices'])])
        torch.save(dict(behaviour={k:v.cpu() for k,v in behaviour.items()},
            demonstration={k:v.cpu() for k,v in demonstration.items()}, counts=counts.cpu()),folder/'scores.pt')
        for name,value in [('W1',w1),('W2',w2)]: atomic_json(folder/f'{name}_selection.json',value)
        atomic_json(folder/'consistency.json',consistency)
        atomic_json(folder/'COMPLETE.json',dict(schema='width-calibration-v1',passed=True,
            teacher=json.loads((root/'teacher_identity.json').read_text()),
            implementation_hashes=implementation_hashes(),configuration_sha256=file_sha(args.config),panel_sha256=file_sha(folder/'panel.json'),
            scores_sha256=file_sha(folder/'scores.pt'),
            selections={name:file_sha(folder/f'{name}_selection.json') for name in ('W1','W2')},
            all_one_gates_exact_on_all_ranks=True,action_sha256=action_sha,
            native_parameters_frozen=True,windows=384,noise_draws_per_window=2,probes_per_draw=2,
            zero_sensitivity_ffn_layers=w2['zero_sensitivity_ffn_layers'],consistency=consistency,
            seconds=time.monotonic()-start))
    for handle in handles: handle.remove()
    dist.barrier(); dist.destroy_process_group()

if __name__=='__main__': main()
