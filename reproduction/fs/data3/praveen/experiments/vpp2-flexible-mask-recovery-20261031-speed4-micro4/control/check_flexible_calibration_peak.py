"""Disposable full calibration update: no production state is written."""
import argparse, copy, json, math, time
from pathlib import Path
import torch
import torch.distributed as dist
from vpp2.flexible_mask.config import load,identity
from vpp2.flexible_mask.runtime import Runtime,distributed
from vpp2.flexible_mask.continuation import load_parent
from vpp2.flexible_mask.data import stream,positions
from vpp2.flexible_mask.controller import schedule
from vpp2.flexible_mask.train import mean_distributed,learning_rate_fraction
from vpp2.flexible_mask.preflight import frozen_fingerprint,fingerprint
from vpp2.compact_kv.spec import atomic_json
from vpp2.full_mask.panel import file_hash
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--candidate',choices=['optimized','micro4'],required=True);a=p.parse_args()
config=load(a.config);execution=copy.deepcopy(config)
if a.candidate=='micro4':execution['train']['micro_batch']=4
rank,world,_=distributed(execution);runtime=Runtime(execution)
state=load_parent(runtime,config,'mask');assert state['update']==47
runtime.engine.train();frozen_before=frozen_fingerprint(runtime)
panel=json.loads(Path(config['panel_file']).read_text());samples=stream(panel,config['seed'])
update=193;batches=positions(update,rank,world,execution['train']['micro_batch']);accum=len(batches)
for group in runtime.engine.optimizer.param_groups:group['lr']=1e-6*learning_rate_fraction(update,3000,50)
for group in runtime.mask_optimizer.param_groups:group['lr']=.01*learning_rate_fraction(update,3000,50)
gt=[torch.zeros_like(p) for p in runtime.masks.families()]
aux={k:[torch.zeros_like(p) for p in runtime.masks.families()] for k in ('video','kv')};size_vectors=None
start=time.monotonic();release_statistics=[]
for pos in batches:
    rows=[samples.at(p) for p in pos]
    loss,metrics,details,terms=runtime.forward(rows,pos,update,measure_gt=True)
    size=runtime.cost.loss(runtime.masks.gates(update));entropy,discrete=runtime.masks.regularizers(update);s=schedule(update)
    for out,g in zip(gt,runtime.measured_mask_gradients['gt']):out.add_(g/accum)
    for key in aux:
        for out,g in zip(aux[key],runtime.measured_mask_gradients[key]):out.add_(g/accum)
    if size_vectors is None:size_vectors=[g.detach() for g in torch.autograd.grad(size,runtime.masks.families(),retain_graph=True)]
    loss=loss+s['entropy']*entropy+s['discreteness']*discrete
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite calibration objective')
    from vpp2.flexible_mask.calibration_batch import release_unused_cache
    release_statistics.append(release_unused_cache(runtime.device))
    runtime.engine.backward(loss/accum,scale_wrt_gas=False);runtime.check_frozen()
    boundary=runtime.engine.is_gradient_accumulation_boundary()
    if boundary:
        for param in runtime.masks.families():
            assert param.grad is not None and torch.isfinite(param.grad).all()
            mean_distributed(param.grad)
        torch.nn.utils.clip_grad_norm_(runtime.masks.parameters(),1.,error_if_nonfinite=True)
    runtime.engine.step()
    if boundary:runtime.mask_optimizer.step();runtime.mask_optimizer.zero_grad(set_to_none=True)
    del loss,metrics,details,terms
norms=dict(gt=[float(mean_distributed(g).norm()) for g in gt],size=[float(mean_distributed(g).norm()) for g in size_vectors])
for key in aux:norms[key]=[float(mean_distributed(g).norm()) for g in aux[key]]
assert all(math.isfinite(x) for values in norms.values() for x in values)
assert all(x>1e-12 for x in norms['gt'])
assert runtime.engine.global_steps==48
assert frozen_fingerprint(runtime)==frozen_before
peak=torch.cuda.max_memory_allocated()/2**30
assert peak<=76,peak
report=dict(identity=identity(config),variant=a.candidate,rank=rank,micro_batch=execution['train']['micro_batch'],
  global_batch=256,passed=True,complete_forward_backward_optimizer_update=True,calibration_position_update=193,
  parent_update=47,disposable_optimizer_update=48,gt_gradients_active=True,gradient_norms=norms,
  frozen_unchanged=True,max_allocated_gib=peak,seconds=time.monotonic()-start,
  release_statistics=release_statistics,script_sha256=file_hash(__file__),execution=execution['execution'],production_weights_written=False)
folder=Path(config['root'])/'speed_review'/a.candidate/'full_calibration';folder.mkdir(parents=True,exist_ok=True)
atomic_json(folder/f'rank{rank}.json',report)
dist.barrier()
if rank==0:print(json.dumps(report),flush=True)
