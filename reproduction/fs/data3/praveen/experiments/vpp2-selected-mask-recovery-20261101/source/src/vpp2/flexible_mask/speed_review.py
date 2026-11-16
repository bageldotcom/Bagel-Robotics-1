"""Matched, disposable GPU benchmarks; never write production checkpoints."""
import argparse
import copy
import gc
import json
import math
import statistics
import time
from pathlib import Path
import torch
import torch.distributed as dist
from .config import load,identity
from .continuation import load_parent
from .runtime import Runtime,distributed
from .data import stream,positions,prepare_cached
from .metrics import DeferredMetrics
from .controller import schedule
from .coefficient import Coefficient
from .train import learning_rate_fraction,mean_distributed
from .preflight import gpu_contracts,frozen_fingerprint,runtime_fingerprint,require_receipt,fingerprint
from .rng import diagnostic_rng
from vpp2.compact_kv.spec import atomic_json
from vpp2.full_mask.panel import file_hash


def run(config,variant,updates=3):
    if variant not in ('baseline','optimized','micro4','logging'):raise ValueError('Unknown benchmark')
    require_receipt(config,'cpu')
    execution=copy.deepcopy(config)
    if variant in ('baseline','logging'):
        execution['train']['micro_batch']=2
        execution['execution'].update(cached_norm_flags=False,skip_nograd_checkpointing=False,
                                      mixed_attention_checkpointing=True,deduplicate_conditioning_reads=False)
    if variant=='micro4':execution['train']['micro_batch']=4
    rank,world,_=distributed(execution)
    runtime=Runtime(execution)
    panel=json.loads(Path(config['panel_file']).read_text())
    contracts=gpu_contracts(runtime,panel)
    state=load_parent(runtime,config,'mask');start=state['update']
    coefficient=Coefficient.from_state(state['coefficient'],config['size_coefficient']['upper_cap'])
    if runtime.engine.train_micro_batch_size_per_gpu()!=execution['train']['micro_batch']:
        raise RuntimeError('Checkpoint load changed execution batch size')
    samples=stream(panel,config['seed'])
    folder=Path(config['root'])/'speed_review'/variant;folder.mkdir(parents=True,exist_ok=True)
    frozen_before=frozen_fingerprint(runtime)
    initial=runtime_fingerprint(runtime)
    # Warm precisely the compared observation encodings before BOTH variants.
    # This prevents a later candidate from winning only because the baseline
    # populated the frozen encoder cache. Noise remains position-addressed.
    from vpp2.full_mask.panel import seed_for
    with diagnostic_rng():
        for u in range(start+1,start+updates+1):
            for pos in positions(u,rank,world,execution['train']['micro_batch']):
                rows=[samples.at(p) for p in pos]
                strata=[seed_for(config['seed'],'stratum',p)%4 for p in pos]
                prepared=prepare_cached(runtime.reference,runtime.reader,rows,pos,config['seed'],strata)
                del prepared
    torch.cuda.synchronize()
    parent_masks={k:v.detach().cpu().clone() for k,v in runtime.masks.state_dict().items()}
    records=[];all_predictions=[];mask_gradients=[]
    runtime.engine.train()
    # Capture action outputs only (small tensors); teacher/student see the same
    # position-addressed noise. No forward/backward arithmetic is replaced.
    native_predict=runtime.predict
    captured=[]
    def predict(parts,cache):
        result=native_predict(parts,cache)
        captured.append(result.detach())
        return result
    runtime.predict=predict
    for update in range(start+1,start+updates+1):
        tick=time.monotonic();rows_metrics=[];predictions=[];captured.clear();collected=DeferredMetrics(runtime.device)
        lr=learning_rate_fraction(update,3000,50)
        for group in runtime.engine.optimizer.param_groups:group['lr']=1e-6*lr
        for group in runtime.mask_optimizer.param_groups:group['lr']=.01*lr
        batches=positions(update,rank,world,execution['train']['micro_batch']);accum=len(batches)
        sample_proofs=[]
        for pos in batches:
            rows=[samples.at(p) for p in pos]
            captured.clear()
            loss,metrics,details,terms=runtime.forward(rows,pos,update)
            if len(captured)!=2:raise RuntimeError('Unexpected action forward inventory')
            for j,p in enumerate(pos):
                predictions.append((p,captured[0][j],captured[1][j],details['reliability'][j].detach()))
                sample_proofs.append((p,rows[j]['id'],rows[j]['task'],rows[j]['frame']))
            size=runtime.cost.loss(runtime.masks.gates(update));entropy,discrete=runtime.masks.regularizers(update);s=schedule(update)
            loss=loss+coefficient.weight(update)*size+s['entropy']*entropy+s['discreteness']*discrete
            runtime.engine.backward(loss/accum,scale_wrt_gas=False)
            runtime.check_frozen();boundary=runtime.engine.is_gradient_accumulation_boundary()
            if boundary:
                for p in runtime.masks.families():
                    if p.grad is None or not torch.isfinite(p.grad).all():raise RuntimeError('Invalid mask gradients')
                    mean_distributed(p.grad)
                grads=[p.grad.detach().clone() for p in runtime.masks.families()]
                torch.nn.utils.clip_grad_norm_(runtime.masks.parameters(),1.,error_if_nonfinite=True)
            runtime.engine.step()
            if boundary:runtime.mask_optimizer.step();runtime.mask_optimizer.zero_grad(set_to_none=True)
            metrics['total']=loss.detach()
            if variant=='baseline':
                keys=list(metrics)
                vals=torch.stack([torch.as_tensor(metrics[k],device=runtime.device).detach().float() for k in keys]).cpu().tolist()
                rows_metrics.append(dict(zip(keys,vals)))
                # The original loop also materializes per-window task metrics.
                _=torch.stack([details[k] for k in ('student_error','teacher_error','reliability')],-1).cpu().tolist()
            else:collected.append(metrics,details,rows)
            del loss,metrics,details,terms,captured[:]
        if variant=='baseline':values={k:sum(x[k] for x in rows_metrics)/len(rows_metrics) for k in rows_metrics[0]}
        else:values,_=collected.finish()
        torch.cuda.synchronize();elapsed=time.monotonic()-tick
        predictions=[(p,t.cpu(),a.cpu(),r.cpu()) for p,t,a,r in predictions]
        grads=[g.cpu() for g in grads]
        keys=sorted(values);means=mean_distributed(torch.tensor([values[k] for k in keys],device=runtime.device,dtype=torch.float64)).tolist()
        values=dict(zip(keys,means))
        durations=mean_distributed(torch.tensor(elapsed,device=runtime.device,dtype=torch.float64))
        values.update(update=update,seconds_per_update=float(durations),video_gradient_norm=float(runtime.engine.get_global_grad_norm()))
        items=[None]*world;dist.all_gather_object(items,predictions)
        proofs=[None]*world;dist.all_gather_object(proofs,sample_proofs)
        proofs=sorted(x for part in proofs for x in part)
        if [x[0] for x in proofs]!=list(range((update-1)*256,update*256)):raise RuntimeError('Incomplete global batch')
        values['samples_sha256']=fingerprint(proofs)
        if rank==0:
            all_predictions.append(sorted((x for part in items for x in part),key=lambda x:x[0]));mask_gradients.append(grads)
            records.append(values);atomic_json(folder/'metrics.json',records)
            print(json.dumps({'variant':variant,**{k:values[k] for k in ('update','seconds_per_update','gt_action','video_gradient_norm')}}),flush=True)
    runtime.predict=native_predict
    if variant!='baseline':
        # Exercise the real calibration graph at its first scheduled update,
        # with two windows on every rank, including all three mask derivatives.
        before=runtime_fingerprint(runtime)
        pos=positions(193,rank,world,execution['train']['micro_batch'])[0];rows=[samples.at(p) for p in pos]
        with diagnostic_rng():
            loss,metrics,details,terms=runtime.forward(rows,pos,193,measure_gt=True)
            calibration={k:[float(g.norm()) for g in gradients] for k,gradients in runtime.measured_mask_gradients.items()}
            if any(not math.isfinite(x) for norms in calibration.values() for x in norms):raise RuntimeError('Nonfinite calibration gradient')
            if any(x<=1e-12 for x in calibration['gt']):raise RuntimeError('Inactive calibration GT gradient')
            del loss,metrics,details,terms
        gc.collect();torch.cuda.synchronize()
        after=runtime_fingerprint(runtime)
        if before!=after:raise RuntimeError('Calibration changed model/optimizer/masks/RNG')
    else:calibration=None
    if frozen_before!=frozen_fingerprint(runtime):raise RuntimeError('Frozen reference changed')
    final=runtime_fingerprint(runtime)
    if initial['video']==final['video'] or initial['masks']==final['masks']:raise RuntimeError('Video/masks failed to learn')
    report=dict(identity=identity(config),variant=variant,start=start,end=start+updates,contracts=contracts,
        micro_batch=execution['train']['micro_batch'],execution=execution['execution'],global_batch=256,initial=initial,final=final,
        frozen_unchanged=True,calibration=calibration,
        max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,max_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        cache_batch_statistics=getattr(runtime.reader,'batch_cache_statistics',{}))
    atomic_json(folder/f'rank{rank}.json',report)
    if rank==0:
        torch.save(dict(predictions=all_predictions,mask_gradients=mask_gradients,initial_masks=parent_masks),folder/'comparison_tensors.pt')
    dist.barrier()


def relative_rmse(left,right):
    return float((left.double()-right.double()).square().mean().sqrt()/left.double().square().mean().sqrt().clamp_min(1e-12))



def benchmark_reference(config):
    root=Path(config['root']);p=root/'control/benchmark_reference.json'
    if not p.exists():return root,identity(config)
    reference=json.loads(p.read_text());previous=Path(reference['root'])
    for name,sha in reference['pinned_files'].items():
        if file_hash(previous/name)!=sha:raise ValueError('Benchmark reference changed: '+name)
    old_manifest=json.loads((previous/'source_manifest.json').read_text())
    current=json.loads((root/'source_manifest.json').read_text())
    changes={name for name in old_manifest['files'].keys()|current['files'].keys()
             if old_manifest['files'].get(name)!=current['files'].get(name)}
    allowed={'src/vpp2/flexible_mask/calibration_batch.py','src/vpp2/flexible_mask/runtime.py',
             'src/vpp2/flexible_mask/config.py','src/vpp2/flexible_mask/continuation.py',
             'src/vpp2/flexible_mask/speed_review.py','tests/flexible_mask/test_calibration_batch.py'}
    if not changes<=allowed:raise ValueError('Uncovered benchmark source changes: '+str(changes))
    from .continuation import scientific_recipe
    if scientific_recipe(json.loads((previous/'config.json').read_text()))!=scientific_recipe(config):
        raise ValueError('Benchmark scientific recipe changed')
    return previous,reference['identity']


def compare(config,variant):
    root=Path(config['root']);folder=root/'speed_review';ident=identity(config)
    reference,reference_identity=benchmark_reference(config)
    names=('baseline',variant)
    folders={'baseline':reference/'speed_review/baseline',variant:folder/variant}
    reports={name:[json.loads((folders[name]/f'rank{r}.json').read_text()) for r in range(8)] for name in names}
    records={name:json.loads((folders[name]/'metrics.json').read_text()) for name in names}
    tensors={name:torch.load(folders[name]/'comparison_tensors.pt',map_location='cpu',weights_only=True) for name in names}
    failures=[];exact=True;numeric=[]
    for rank,(a,b) in enumerate(zip(reports['baseline'],reports[variant])):
        if a['identity']!=reference_identity or b['identity']!=ident or (a['start'],a['end'])!=(b['start'],b['end']):failures.append('identity/update')
        if a['initial']!=b['initial']:failures.append(f'rank{rank}/initial-state')
        if not b['frozen_unchanged'] or not b['calibration']:failures.append(f'rank{rank}/frozen-or-calibration')
        if a['final']!=b['final']:exact=False
    for i,(a,b) in enumerate(zip(records['baseline'],records[variant],strict=True)):
        if a['samples_sha256']!=b['samples_sha256']:failures.append('sample-sequence')
        left=tensors['baseline']['predictions'][i];right=tensors[variant]['predictions'][i]
        if [r[0] for r in left]!=[r[0] for r in right]:failures.append('prediction-order')
        item=dict(update=a['update'])
        for index,key in ((1,'teacher_action'),(2,'student_action')):
            x=torch.stack([r[index] for r in left]);y=torch.stack([r[index] for r in right])
            item[key+'_relative_rmse']=relative_rmse(x,y)
            exact=exact and torch.equal(x,y)
        x=torch.stack([r[3] for r in left]);y=torch.stack([r[3] for r in right])
        item['reliability_mean_absolute_difference']=float((x-y).abs().mean())
        item['reliability_rejection_flip_fraction']=float(((x==0)!=(y==0)).float().mean())
        for key in ('gt_action','teacher_video_gated','late_kv_gated','total','video_gradient_norm'):
            item[key+'_relative_difference']=abs(a[key]-b[key])/max(abs(a[key]),1e-8)
            exact=exact and a[key]==b[key]
        for family,ga,gb in zip(('hidden','ffn','cross'),tensors['baseline']['mask_gradients'][i],tensors[variant]['mask_gradients'][i]):
            item[family+'_gradient_cosine']=float(torch.nn.functional.cosine_similarity(ga.float().flatten(),gb.float().flatten(),dim=0))
            item[family+'_gradient_norm_relative_difference']=abs(float(ga.norm())-float(gb.norm()))/max(float(ga.norm()),1e-12)
            exact=exact and torch.equal(ga,gb)
        numeric.append(item)
    base=statistics.mean(x['seconds_per_update'] for x in records['baseline'][1:])
    speed=statistics.mean(x['seconds_per_update'] for x in records[variant][1:])
    gain=base/speed
    # Relax BF16 batching tolerance only for a measured substantial gain.
    relaxed=reports[variant][0]['micro_batch']!=2 and gain>=1.20
    output_limit=.02 if relaxed else .01
    grad_limit=.10 if relaxed else .05
    for item in numeric:
        for key,value in item.items():
            if not math.isfinite(value):failures.append('nonfinite/'+key)
            if key.endswith('_relative_rmse') and value>output_limit:failures.append(key)
            if key.endswith('_relative_difference'):
                limit=grad_limit if ('gradient' in key) else output_limit
                if value>limit:failures.append(key)
            if key.endswith('_gradient_cosine') and value<.99:failures.append(key)
        if item['reliability_rejection_flip_fraction']>.02 or item['reliability_mean_absolute_difference']>.02:failures.append('reliability-change')
    # Same-batch execution-only candidates should be exact, not exploit the
    # relaxed tolerance intended for a different BF16 microbatch layout.
    if reports[variant][0]['micro_batch']==2 and not exact:failures.append('same-microbatch-state-not-exact')
    peak=max(x['max_allocated_gib'] for x in reports[variant])
    if peak>76:failures.append('memory-headroom')
    numerics_passed=not failures
    if gain<1.02:failures.append('insufficient-speedup')
    report=dict(identity=ident,reference_identity=reference_identity,reference_root=str(reference),variant=variant,passed=not failures,numerics_passed=numerics_passed,failures=failures,
        baseline_seconds=base,candidate_seconds=speed,speedup=gain,peak_allocated_gib=peak,
        exact_final_state_and_numerics=exact,relaxed_bf16_tolerance_used=relaxed,output_tolerance=output_limit,
        gradient_norm_tolerance=grad_limit,gradient_cosine_minimum=.99,numerics=numeric,
        micro_batch=reports[variant][0]['micro_batch'],execution=reports[variant][0]['execution'],
        checkpoint=config['resume_from'],scientific_objectives_unchanged=True,
        timing_excludes_cold_first_update=True,frozen_conditioning_cache_prewarmed_for_both=True,
        evidence={str(p.resolve()):file_hash(p) for name in names for p in sorted(folders[name].glob('*')) if p.is_file()})
    atomic_json(root/'checks'/('speed_'+variant+'.json'),report)
    print(json.dumps(report),flush=True)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--variant',choices=['baseline','optimized','micro4','logging','compare'],required=True)
    p.add_argument('--candidate',choices=['optimized','micro4','logging']);p.add_argument('--updates',type=int,default=3)
    a=p.parse_args();c=load(a.config)
    if a.variant=='compare':
        if not a.candidate:raise ValueError('Choose a completed candidate')
        compare(c,a.candidate)
    else:run(c,a.variant,a.updates)

if __name__=='__main__':main()
