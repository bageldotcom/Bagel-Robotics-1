"""Matched checkpoint-recomputation benchmark; never trains or writes readiness."""
import argparse
import gc
import json
import time
from pathlib import Path
from types import MethodType
import torch
from torch.utils.checkpoint import checkpoint
from vpp2.adaptive_mask.runtime import distributed, initialize, step, model_versions, check_frozen
from vpp2.adaptive_mask.data import prepare
from vpp2.adaptive_mask.state import identity


def selective_forward(self, inputs, noise, timestep, video_mask, hidden_gate=None, ffn_gates=None,
                      cross_gates=None, *, return_video=True):
    pre = self.video.pre_dit(noise, timestep, inputs['context'], inputs['context_mask'],
                             inputs['clip_fea'], inputs['condition_latents'], hidden_gate=hidden_gate)
    x = pre['tokens']; caches = []
    for i, block in enumerate(self.video.blocks):
        f = None if ffn_gates is None or i == len(self.video.blocks)-1 else ffn_gates[i]
        c = None if cross_gates is None or i == len(self.video.blocks)-1 else cross_gates[i]
        args = (x, pre['t_mod'], pre['freqs'], pre['context'], pre['context_mask'], video_mask, hidden_gate, f, c)
        if self.training and torch.is_grad_enabled() and i in self.checkpoint_layers:
            x, k, v = checkpoint(block, *args, use_reentrant=False)
        else:
            x, k, v = block(*args)
        caches.append(dict(k=k, v=v))
    if not return_video: return caches
    video = self.video.unpatchify(self.video.head(x, pre['t'], hidden_gate), pre['meta']['grid_size'])
    return dict(cache=caches, video=video)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); config = json.loads(Path(args.config).read_text())
    panel = json.loads(Path(config['panel_file']).read_text()); distributed()
    model, reader, controller, branch, inventory = initialize(config); versions = model_versions(model)
    row = next(r for r in panel['windows'] if r['role']=='selection_initial' and r['suite']=='precision')
    parts = prepare(model, reader, [row], [12345], config['seed'], [2])
    controller.project(inventory, inventory.full_at_hidden(4864)-250_000_000, 4864)
    original_forward = branch.forward
    candidates = [('baseline', list(range(40)), True), ('action_no_recompute', list(range(40)), False),
                  ('video_half_recompute', list(range(0,40,2)), False),
                  ('video_quarter_recompute', list(range(0,40,4)), False), ('no_recompute', [], False)]
    report = dict(identity=identity(config,panel), diagnostic_only=True, micro_batch=1, candidates=[], weights_updated=False)
    reference = None
    def trial():
        controller.zero_grad(set_to_none=True); torch.cuda.synchronize()
        start = time.perf_counter()
        loss, metrics, _, _ = step(model, branch, controller, parts, [row], 100, config)
        loss.backward(); torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        values = {k:float(v) for k,v in metrics.items() if k.startswith('loss/') or k.startswith('reliability/')}
        gradients = torch.cat([p.grad.detach().float().flatten() for p in controller.trainable_parameters()]).cpu()
        if not torch.isfinite(gradients).all(): raise FloatingPointError('Nonfinite score gradients')
        return seconds, values, gradients
    for name, layers, action_checkpoint in candidates:
        record = dict(name=name, video_checkpoint_layers=layers, action_checkpoint=action_checkpoint)
        model.action_expert.use_gradient_checkpointing = action_checkpoint
        branch.forward = original_forward if name=='baseline' else MethodType(selective_forward, branch)
        branch.checkpoint_layers = set(layers)
        try:
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            trial()  # allocator/kernel warm-up, no optimizer update
            samples = [trial(), trial()]
            seconds, values, gradients = samples[-1]
            if reference is None: reference = (values, gradients)
            gradient_relative = float((gradients-reference[1]).square().mean().sqrt()/reference[1].square().mean().sqrt().clamp_min(1e-12))
            loss_absolute = max(abs(values[k]-reference[0][k]) for k in values)
            record.update(completed=True,seconds_per_microbatch=sum(s[0] for s in samples)/len(samples),
                          gradient_relative_rmse=gradient_relative, gradient_bitwise_equal=torch.equal(gradients,reference[1]),
                          max_loss_or_reliability_absolute_difference=loss_absolute,
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                          gradient_and_loss_match=gradient_relative<=1e-5 and loss_absolute<=1e-6)
            del samples, gradients
        except torch.OutOfMemoryError:
            record.update(completed=False,reason='cuda_out_of_memory')
        finally:
            controller.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
        check_frozen(model,versions)
        report['candidates'].append(record)
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(record),flush=True)
    print('Benchmark complete; no optimization has been applied to queued training.',flush=True)


if __name__=='__main__': main()
