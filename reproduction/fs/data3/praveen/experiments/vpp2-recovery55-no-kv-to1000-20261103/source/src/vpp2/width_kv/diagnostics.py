"""Independent training-only checks of action-sensitivity ranking and sliced models."""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import default_collate
from vpp2.compact_kv.blocks import install_blocks
from vpp2.compact_kv.model import load_reference, load_student, VideoBranch, action_prediction
from vpp2.compact_kv.losses import action_error, kv_terms
from vpp2.compact_kv.spec import atomic_json, file_sha
from .common import load_panel, prepare_calibration, implementation_hashes
from .scoring import install_gates, task_scores


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True)
    parser.add_argument('--mode',choices=['sensitivity','W1','W2'],required=True); args=parser.parse_args()
    config=json.loads(Path(args.config).read_text()); root=Path(config['root']); folder=root/'calibration'
    rank=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=timedelta(minutes=40)); world=dist.get_world_size()
    if world != 8: raise RuntimeError('Expected eight ranks')
    torch.manual_seed(config['seed']); torch.backends.cuda.matmul.allow_tf32=False
    receipt=json.loads((folder/'COMPLETE.json').read_text())
    if receipt['implementation_hashes'] != implementation_hashes() or receipt['configuration_sha256'] != file_sha(args.config): raise RuntimeError('Stale calibration')
    dataset,panel=load_panel(config)
    reference=load_reference(config,device=f'cuda:{rank}'); reference.mot.training=False
    install_blocks(reference.video_expert,()); teacher=VideoBranch(reference.video_expert).eval()
    if args.mode == 'sensitivity':
        scores=torch.load(folder/'scores.pt',map_location='cpu',weights_only=True)
        importance=task_scores(scores['behaviour']['residual'].mean((0,2)),scores['demonstration']['residual'].mean((0,2)))
        order=torch.argsort(importance,descending=True,stable=True)
        high,low=order[:64].to(f'cuda:{rank}'),order[-64:].to(f'cuda:{rank}')
        residual,ffn,handles=install_gates(reference.video_expert)
    else:
        student=load_student(config,args.mode).to(device=f'cuda:{rank}',dtype=torch.bfloat16).eval()
    entries=[e for e in panel['entries'] if e['partition']=='diagnostic']
    rows=[]
    with torch.no_grad():
        for n,entry in enumerate(entries[rank::world]):
            sample=default_collate([dataset._get(entry['index'])])
            for draw in range(2):
                inputs,noise,tv,noisy,ta,target,mask=prepare_calibration(reference,sample,entry,draw,config)
                tc=teacher(inputs,noise,tv,mask[:1950,:1950])
                baseline=action_prediction(reference,inputs,tc,noisy,ta,mask)
                amask=inputs.get('action_loss_is_pad',inputs.get('action_is_pad'))
                weight=reference.train_action_scheduler.training_weight(ta)
                row=dict(id=entry['id'],task=entry['task'],draw=draw,
                    teacher_flow_gt=float((action_error(baseline,target,amask)*weight).mean()))
                if args.mode=='sensitivity':
                    del tc
                    for name,indices in [('high',high),('low',low)]:
                        for gate in residual:
                            gate.multiplier.fill_(1); gate.multiplier[indices]=0.95
                        cache=teacher(inputs,noise,tv,mask[:1950,:1950])
                        pred=action_prediction(reference,inputs,cache,noisy,ta,mask)
                        row[name+'_action_change']=float(action_error(pred,baseline,amask).mean())
                        row[name+'_flow_gt']=float((action_error(pred,target,amask)*weight).mean())
                        del cache,pred
                    for gate in residual: gate.multiplier.fill_(1)
                else:
                    cache=student(inputs,noise,tv,mask[:1950,:1950])
                    pred=action_prediction(reference,inputs,cache,noisy,ta,mask)
                    row.update(student_flow_gt=float((action_error(pred,target,amask)*weight).mean()),
                        action_teacher=float((action_error(pred,baseline,amask)*weight).mean()),
                        kv_by_layer_kv=kv_terms(cache,tc).mean(0).cpu().tolist())
                    del cache,pred,tc
                rows.append(row)
                del inputs,noise,tv,noisy,ta,target,mask,baseline
            if n%4==0: print(json.dumps(dict(event='width_diagnostic',mode=args.mode,rank=rank,done=n+1)),flush=True)
    gathered=[None]*world; dist.all_gather_object(gathered,rows)
    if rank==0:
        rows=sorted([x for part in gathered for x in part],key=lambda x:(x['id'],x['draw']))
        if len(rows)!=256 or len({(x['id'],x['draw']) for x in rows})!=256: raise RuntimeError('Missing/duplicate diagnostics')
        def finite(value):
            if isinstance(value,float) and not math.isfinite(value): raise FloatingPointError('Nonfinite calibration diagnostic')
            if isinstance(value,dict):
                for item in value.values(): finite(item)
            elif isinstance(value,list):
                for item in value: finite(item)
        finite(rows)
        keys=[k for k,v in rows[0].items() if isinstance(v,float)]
        summary={task:{k:sum(r[k] for r in rows if r['task']==task)/32 for k in keys} for task in sorted({r['task'] for r in rows})}
        result=dict(passed=True,mode=args.mode,windows=128,draws=2,rows=rows,by_task=summary,
            implementation_hashes=implementation_hashes(),calibration_sha256=file_sha(folder/'COMPLETE.json'))
        if args.mode=='sensitivity':
            result['high_more_sensitive_fraction']=sum(r['high_action_change']>r['low_action_change'] for r in rows)/len(rows)
            result['interpretation']='Diagnostic evidence only; a low ranking signal is reported, not hidden or automatically tuned.'
        atomic_json(folder/f'diagnostic_{args.mode}.json',result)
    dist.barrier(); dist.destroy_process_group()

if __name__=='__main__': main()
