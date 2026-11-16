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
from .data import stream,positions
from .controller import schedule
from .coefficient import Coefficient
from .train import learning_rate_fraction,mean_distributed
from .preflight import gpu_contracts,frozen_fingerprint,runtime_fingerprint,require_receipt,fingerprint
from .rng import diagnostic_rng
from vpp2.compact_kv.spec import atomic_json
from vpp2.full_mask.panel import file_hash


def run(config,variant,updates=3):
    if variant not in ('baseline','micro2'):raise ValueError('Unknown benchmark')
    require_receipt(config,'cpu')
    execution=copy.deepcopy(config)
    if variant=='baseline':
        execution['train']['micro_batch']=1
        execution['execution']['fill_uninitialized_memory']=True
    rank,world,_=distributed(execution)
    runtime=Runtime(execution)
    panel=json.loads(Path(config['panel_file']).read_text())
    contracts=gpu_contracts(runtime,panel)
    state=load_parent(runtime,config,'mask');start=state['update']
    coefficient=Coefficient.from_state(state['coefficient'],config['size_coefficient']['upper_cap'])
    if runtime.engine.train_micro_batch_size_per_gpu()!=execution['train']['micro_batch']:
        raise RuntimeError('Checkpoint load changed execution batch size')
    samples=stream(panel,config['seed'])
    folder=Path(config['root'])/'throughput'/variant;folder.mkdir(parents=True,exist_ok=True)
    frozen_before=frozen_fingerprint(runtime)
    initial=runtime_fingerprint(runtime)
    parent_masks={k:v.detach().cpu().clone() for k,v in runtime.masks.state_dict().items()}
    records=[];all_predictions=[];mask_gradients=[]
    runtime.engine.train()
    # Capture action outputs only (small tensors); teacher/student see the same
    # position-addressed noise. No forward/backward arithmetic is replaced.
    native_predict=runtime.predict
    captured=[]
    def predict(parts,cache):
        result=native_predict(parts,cache)
        captured.append(result.detach().cpu())
        return result
    runtime.predict=predict
    for update in range(start+1,start+updates+1):
        tick=time.monotonic();rows_metrics=[];predictions=[];captured.clear()
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
                predictions.append((p,captured[0][j],captured[1][j],details['reliability'][j].detach().cpu()))
                sample_proofs.append((p,rows[j]['id'],rows[j]['task'],rows[j]['frame']))
            size=runtime.cost.loss(runtime.masks.gates(update));entropy,discrete=runtime.masks.regularizers(update);s=schedule(update)
            loss=loss+coefficient.weight(update)*size+s['entropy']*entropy+s['discreteness']*discrete
            runtime.engine.backward(loss/accum,scale_wrt_gas=False)
            runtime.check_frozen();boundary=runtime.engine.is_gradient_accumulation_boundary()
            if boundary:
                for p in runtime.masks.families():
                    if p.grad is None or not torch.isfinite(p.grad).all():raise RuntimeError('Invalid mask gradients')
                    mean_distributed(p.grad)
                grads=[p.grad.detach().cpu().clone() for p in runtime.masks.families()]
                torch.nn.utils.clip_grad_norm_(runtime.masks.parameters(),1.,error_if_nonfinite=True)
            runtime.engine.step()
            if boundary:runtime.mask_optimizer.step();runtime.mask_optimizer.zero_grad(set_to_none=True)
            metrics['total']=loss.detach()
            keys=list(metrics)
            vals=torch.stack([torch.as_tensor(metrics[k],device=runtime.device).detach().float() for k in keys]).cpu().tolist()
            rows_metrics.append(dict(zip(keys,vals)))
            del loss,metrics,details,terms,captured[:]
        torch.cuda.synchronize();elapsed=time.monotonic()-tick
        values={k:sum(x[k] for x in rows_metrics)/len(rows_metrics) for k in rows_metrics[0]}
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
    if variant=='micro2':
        # Exercise the real calibration graph at its first scheduled update,
        # with two windows on every rank, including all three mask derivatives.
        before=runtime_fingerprint(runtime)
        pos=positions(193,rank,world,2)[0];rows=[samples.at(p) for p in pos]
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
        micro_batch=execution['train']['micro_batch'],global_batch=256,initial=initial,final=final,
        frozen_unchanged=True,calibration=calibration,
        max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,max_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        cache_batch_statistics=getattr(runtime.reader,'batch_cache_statistics',{}))
    atomic_json(folder/f'rank{rank}.json',report)
    if rank==0:
        torch.save(dict(predictions=all_predictions,mask_gradients=mask_gradients,initial_masks=parent_masks),folder/'comparison_tensors.pt')
    dist.barrier()


def relative_rmse(left,right):
    return float((left.double()-right.double()).square().mean().sqrt()/left.double().square().mean().sqrt().clamp_min(1e-12))


def compare(config):
    root=Path(config['root']);folder=root/'throughput';ident=identity(config)
    reports={name:[json.loads((folder/name/f'rank{r}.json').read_text()) for r in range(8)] for name in ('baseline','micro2')}
    baseline=json.loads((folder/'baseline/metrics.json').read_text());candidate=json.loads((folder/'micro2/metrics.json').read_text())
    failures=[]
    baseline_identity=ident
    reference=root/'checks/throughput_reference.json'
    if reference.exists():
        ref=json.loads(reference.read_text())
        if ref['identity']!=ident:raise ValueError('Stale throughput reference')
        baseline_identity=ref['executed_identity']
        for name,sha in ref['files'].items():
            if file_hash(folder/'baseline'/name)!=sha:raise ValueError('Changed baseline evidence')
        previous=json.loads(Path(ref['source_manifest']).read_text())
        if file_hash(ref['source_manifest'])!=ref['source_manifest_sha256']:raise ValueError('Reference source manifest changed')
        current=json.loads((root/'source_manifest.json').read_text())
        changed={k for k in previous['files'].keys()|current['files'].keys() if previous['files'].get(k)!=current['files'].get(k)}
        allowed={'src/vpp2/flexible_mask/data.py','src/vpp2/flexible_mask/runtime.py','src/vpp2/flexible_mask/throughput.py','tests/flexible_mask/test_throughput_continuation.py'}
        if not changed or not changed<=allowed:raise ValueError('Unverified baseline source changes: '+str(changed))
    for rank in range(8):
        a,b=(reports[n][rank] for n in ('baseline','micro2'))
        if a['identity']!=baseline_identity or b['identity']!=ident or (a['start'],a['end'])!=(b['start'],b['end']):failures.append('identity/update mismatch')
        if a['initial']!=b['initial']:failures.append(f'rank{rank}/initial-state')
        if not b['frozen_unchanged'] or not b['calibration']:failures.append(f'rank{rank}/frozen-calibration')
    tensors={name:torch.load(folder/name/'comparison_tensors.pt',map_location='cpu',weights_only=True) for name in ('baseline','micro2')}
    numeric=[]
    for i,(a,b) in enumerate(zip(baseline,candidate,strict=True)):
        if a['samples_sha256']!=b['samples_sha256']:failures.append('sample-sequence')
        left=tensors['baseline']['predictions'][i];right=tensors['micro2']['predictions'][i]
        if [r[0] for r in left]!=[r[0] for r in right]:failures.append('prediction-order')
        entry=dict(update=a['update'])
        for index,name in ((1,'teacher_action'),(2,'student_action')):
            entry[name+'_relative_rmse']=relative_rmse(torch.stack([r[index] for r in left]),torch.stack([r[index] for r in right]))
            if entry[name+'_relative_rmse']>.01:failures.append(f'update{a["update"]}/{name}')
        for key in ('gt_action','teacher_video_gated','late_kv_gated','total','video_gradient_norm'):
            entry[key+'_relative_difference']=abs(a[key]-b[key])/max(abs(a[key]),1e-8)
            limit=.05 if key=='video_gradient_norm' else .01
            if entry[key+'_relative_difference']>limit:failures.append(f'update{a["update"]}/{key}')
        for family,ga,gb in zip(('hidden','ffn','cross'),tensors['baseline']['mask_gradients'][i],tensors['micro2']['mask_gradients'][i]):
            cos=float(torch.nn.functional.cosine_similarity(ga.float().flatten(),gb.float().flatten(),dim=0))
            rel=abs(float(ga.norm())-float(gb.norm()))/max(float(ga.norm()),1e-12)
            entry[family+'_gradient_cosine']=cos;entry[family+'_gradient_norm_relative_difference']=rel
            if cos<.99 or rel>.05:failures.append(f'update{a["update"]}/{family}-gradient')
        numeric.append(entry)
    def finite(value):
        if isinstance(value,float):return math.isfinite(value)
        if isinstance(value,dict):return all(finite(v) for v in value.values())
        if isinstance(value,list):return all(finite(v) for v in value)
        return True
    if not finite(numeric) or not finite(baseline) or not finite(candidate):failures.append('nonfinite-evidence')
    base_seconds=statistics.mean(x['seconds_per_update'] for x in baseline[1:])
    seconds=statistics.mean(x['seconds_per_update'] for x in candidate[1:])
    peak=max(x['max_allocated_gib'] for x in reports['micro2'])
    if peak>76:failures.append('memory-headroom')
    if base_seconds/seconds<1.10:failures.append('insufficient-speedup')
    report=dict(identity=ident,passed=not failures,failures=failures,baseline_seconds=base_seconds,candidate_seconds=seconds,
        speedup=base_seconds/seconds,peak_allocated_gib=peak,numerics=numeric,
        checkpoint=config['resume_from'],samples_and_initial_state_match=True if not any('initial' in s or 'sequence' in s for s in failures) else False,
        scientific_objectives_unchanged=True,bitwise_equivalent_across_microbatches=False,
        diagnostics='All eight ranks exercised real mask-gradient calibration without optimizer/model/RNG changes',
        compared_updates=len(candidate),timing_discard_first_update=True,baseline_executed_identity=baseline_identity,
        cache_batch_statistics=[r.get('cache_batch_statistics',{}) for r in reports['micro2']])
    atomic_json(root/'checks/throughput.json',report)
    print(json.dumps(report),flush=True)
    if failures:raise RuntimeError('Optimization failed verification: '+str(failures))
    atomic_json(root/'checks/gpu.json',dict(identity=ident,passed=True,world_size=8,micro_batch=2,global_batch=256,
        numerical_comparison_receipt=str(root/'checks/throughput.json'),numerical_comparison_sha256=file_hash(root/'checks/throughput.json'),
        frozen_reference_unchanged=True,contracts=True,diagnostic_gradient_noninterference=True,
        continuation_checkpoint=config['resume_from'],export_equivalence_still_required=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--variant',choices=['baseline','micro2','compare'],required=True)
    a=p.parse_args();c=load(a.config)
    if a.variant=='compare':compare(c)
    else:run(c,a.variant)

if __name__=='__main__':main()
