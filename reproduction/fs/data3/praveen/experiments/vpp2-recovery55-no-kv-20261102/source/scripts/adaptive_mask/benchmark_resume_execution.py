"""Paired full-batch execution benchmark from an immutable training checkpoint.

No training checkpoints or readiness receipts are written. All candidate updates
are discarded. Preparation stays microbatch one to preserve exact sample/noise
and frozen encoder results when testing model microbatch two.
"""
import argparse,copy,gc,hashlib,json,time
from pathlib import Path
import torch
import torch.distributed as dist
from vpp2.adaptive_mask.runtime import distributed,initialize,step,check_frozen,model_versions,gather
from vpp2.adaptive_mask.data import prepare
from vpp2.adaptive_mask.controller import synchronize_controller
from vpp2.adaptive_mask.priority import MaskPriority
from vpp2.full_mask.panel import BalancedStream,noise_stratum
from vpp2.adaptive_mask.state import restore_rng


def stack_parts(parts):
    inputs={}
    for key,value in parts[0][0].items():
        xs=[p[0][key] for p in parts]
        if torch.is_tensor(value):
            if value.ndim and value.shape[0]==1:inputs[key]=torch.cat(xs,0)
            else:
                if not all(torch.equal(x,value) for x in xs):raise ValueError('Nonbatched input differs: '+key)
                inputs[key]=value
        else:
            if not all(x==value for x in xs):raise ValueError('Input scalar differs: '+key)
            inputs[key]=value
    if not all(torch.equal(p[6],parts[0][6]) for p in parts):raise ValueError('Joint attention masks differ')
    return (inputs,*(torch.cat([p[j] for p in parts],0) for j in range(1,6)),parts[0][6])


def tensor_digest(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def flatten_optimizer(value):
    if torch.is_tensor(value):return [value.detach().cpu().clone()]
    if isinstance(value,dict):return sum((flatten_optimizer(value[k]) for k in sorted(value,key=str)),[])
    if isinstance(value,(tuple,list)):return sum((flatten_optimizer(v) for v in value),[])
    return []


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    config=json.loads(Path(a.config).read_text());panel=json.loads(Path(config['panel_file']).read_text());rank,world=distributed()
    if world!=8:raise ValueError('Requires eight allocated GPUs')
    model,reader,controller,branch,inventory=initialize(config);versions=model_versions(model)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    original=copy.deepcopy(saved['state']['controller']);controller.load_state_dict(original)
    restore_rng(saved['rngs'][rank]);cursor=saved['state']['cursor'];pruning=saved['state']['pruning_updates']+1
    fitting=[r for r in panel['windows'] if r['role']=='selection_initial'];stream=BalancedStream(fitting,config['seed'])
    local=[];coverage=[]
    for i in range(32):
        pos=cursor+i*world+rank;row=stream.at(pos);s=noise_stratum(config['seed'],pos,len(panel['tasks']))
        parts=prepare(model,reader,[row],[pos],config['seed'],[s]);local.append((row,parts))
        coverage.append(dict(position=pos,window_id=row['id'],task=row['task'],suite=row['suite'],stratum=int(s),
                             video_noise=tensor_digest(parts[1]),action_noise=tensor_digest(parts[3])))
    coverage=sum(gather(coverage),[])
    if len({x['position'] for x in coverage})!=256 or {x['position'] for x in coverage}!=set(range(cursor,cursor+256)):
        raise AssertionError('Distributed sample sequence changed')
    cases=[dict(name='baseline',cache=False,mixed=True,micro=1,layers=list(range(0,40,2))),
           dict(name='cached_masks',cache=True,mixed=True,micro=1,layers=list(range(0,40,2))),
           dict(name='cached_masks_no_attention_recompute',cache=True,mixed=False,micro=1,layers=list(range(0,40,2))),
           dict(name='micro2_full_recompute',cache=True,mixed=False,micro=2,layers=list(range(40))),
           dict(name='micro2_threequarter_recompute',cache=True,mixed=False,micro=2,layers=[i for i in range(40) if i%4!=3])]
    report=dict(checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
                checkpoint_identity=saved['identity'],cursor=cursor,attempt=saved['budget']['attempt'],global_batch=256,seed=config['seed'],
                diagnostic_only=True,production_weights_or_optimizer_modified=False,candidates=[],coverage=coverage,
                selection_rule='Only bitwise-equal aggregated losses, mask gradients, updated mask scores and optimizer tensors may be promoted.')
    def run(case,state):
        controller.load_state_dict(copy.deepcopy(state));controller.cache_hard_masks=case['cache']
        model.mot.mot_checkpoint_mixed_attn=case['mixed'];model.video_expert.activation_checkpoint_layers=frozenset(case['layers'])
        model.video_expert.use_gradient_checkpointing=bool(case['layers'])
        optimizer=torch.optim.AdamW(controller.trainable_parameters(),lr=config['train']['learning_rate'],betas=tuple(config['train']['betas']),weight_decay=0.)
        optimizer.load_state_dict(copy.deepcopy(saved['state']['optimizer']))
        for group in optimizer.param_groups:group['lr']=config['train']['learning_rate']*min((saved['state']['lineage_update']+1)/config['train']['warmup_updates'],1.)
        priority=MaskPriority(controller,**config['mask_priority']);priority.load_state_dict(saved['priority']);priority.refresh()
        # Kernel warm-up without an optimizer update; no randomness or sample cursor is consumed.
        warm=stack_parts([x[1] for x in local[:case['micro']]]) if case['micro']>1 else local[0][1]
        loss,*_=step(model,branch,controller,warm,[x[0] for x in local[:case['micro']]],pruning,config)
        loss.backward();del loss,_,warm
        optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        begin=time.perf_counter();metrics={};n=32//case['micro']
        for i in range(0,32,case['micro']):
            batch=local[i:i+case['micro']];parts=stack_parts([x[1] for x in batch]) if case['micro']>1 else batch[0][1]
            loss,terms,details,unused=step(model,branch,controller,parts,[x[0] for x in batch],pruning,config)
            (loss/n).backward()
            for key,value in terms.items():
                if key.startswith(('loss/','reliability/','kv/','supervision/')):
                    metrics[key]=metrics.get(key,0)+torch.as_tensor(value,device='cuda').detach()/n
            del loss,terms,details,unused,parts
        torch.cuda.synchronize();seconds=time.perf_counter()-begin
        return optimizer,priority,metrics,seconds
    def measured(case,state,reference=None):
        failure=None;result=None
        try:result=run(case,state)
        except torch.OutOfMemoryError:failure='cuda_out_of_memory'
        failures=gather(failure)
        if any(failures):
            controller.zero_grad(set_to_none=True);result=None;gc.collect();torch.cuda.empty_cache()
            return dict(**case,completed=False,failures=failures),None
        opt,priority,metrics,seconds=result;norm=synchronize_controller(controller,config['train']['gradient_clip'])
        gradients=torch.cat([p.grad.detach().flatten() for p in controller.trainable_parameters()]).cpu().clone()
        priority.step(opt);check_frozen(model,versions)
        scores=torch.cat([p.detach().flatten() for p in controller.trainable_parameters()]).cpu().clone()
        keys=sorted(metrics);values=torch.stack([metrics[k].to(torch.float64) for k in keys]);dist.all_reduce(values);values/=world
        values=values.cpu();optim=flatten_optimizer(opt.state_dict())
        measured=gather(dict(rank=rank,seconds=seconds,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                             peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        tensors=(gradients,scores,values,optim)
        record=dict(**case,completed=True,slowest_seconds=max(x['seconds'] for x in measured),ranks=measured,metrics=dict(zip(keys,values.tolist())),
                    gradient_norm=float(norm))
        if reference is not None:
            g,s,v,o=reference
            record.update(gradient_bitwise_equal=torch.equal(g,gradients),scores_bitwise_equal=torch.equal(s,scores),
                          loss_metrics_bitwise_equal=torch.equal(v,values),optimizer_bitwise_equal=len(o)==len(optim) and all(torch.equal(x,y) for x,y in zip(o,optim)),
                          gradient_relative_rmse=float((gradients-g).square().mean().sqrt()/g.square().mean().sqrt().clamp_min(1e-12)),
                          metrics_max_absolute_difference=float((values-v).abs().max()))
            record['equivalent']=all(record[k] for k in ['gradient_bitwise_equal','scores_bitwise_equal','loss_metrics_bitwise_equal','optimizer_bitwise_equal'])
        else:record['equivalent']=True
        controller.zero_grad(set_to_none=True);del opt,priority,metrics,result;gc.collect();torch.cuda.empty_cache()
        return record,tensors
    def publish():
        if rank==0:Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    ref=None
    for case in cases:
        rec,tensors=measured(case,original,ref)
        if case['name']=='baseline':
            if not rec['completed']:raise RuntimeError('Current verified execution unexpectedly failed')
            ref=tensors
        report['candidates'].append(rec);publish()
        if rank==0:print(json.dumps({k:v for k,v in rec.items() if k not in ['ranks','metrics']}),flush=True)
    valid=[r for r in report['candidates'] if r.get('completed') and r.get('equivalent')]
    chosen=min(valid,key=lambda r:r['slowest_seconds'])
    report['provisional_choice']=chosen['name'];publish()
    # A second comparison stresses 5% hidden pruning and the final 10B budget.
    controller.load_state_dict(original);controller.project(inventory,10_000_000_000,4864);stress=copy.deepcopy(controller.state_dict())
    baseline,stress_ref=measured(cases[0],stress)
    chosen_case=next(x for x in cases if x['name']==chosen['name']);candidate,_=measured(chosen_case,stress,stress_ref)
    report['stress']={'baseline':baseline,'candidate':candidate};report['selected']=chosen['name'] if candidate.get('completed') and candidate.get('equivalent') else 'baseline'
    report['passed']=True;publish()
    if rank==0:print(json.dumps(dict(event='complete',selected=report['selected'],stress_equivalent=candidate.get('equivalent'))),flush=True)
    dist.barrier()

if __name__=='__main__':main()
