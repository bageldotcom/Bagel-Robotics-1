"""Reload a compact native checkpoint without the original video teacher."""
import argparse
import json
from pathlib import Path

import torch
from .model import load_reference, native_cache, action_prediction, infer_from_cache
from .gpu_check import tensor_hash
from .spec import atomic_json, file_sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run',required=True)
    p.add_argument('--update',type=int,required=True);args=p.parse_args()
    config=json.loads(Path(args.config).read_text())
    run=Path(args.run);export=run/f'exports/update_{args.update:06d}'
    complete=json.loads((export/'COMPLETE.json').read_text())
    for name, sha in complete['files'].items():
        if file_sha(export/name)!=sha:raise RuntimeError(f'Export changed: {name}')
    config['teacher_bundle']=str(export)
    model=load_reference(config)  # Loads compact export video.pt, not the original teacher.
    fixture=torch.load(run/f'validation/fixture_{args.update:06d}.pt',map_location='cpu',weights_only=True)
    def gpu(value):
        if isinstance(value,torch.Tensor):return value.to('cuda')
        if isinstance(value,dict):return {k:gpu(v) for k,v in value.items()}
        return value
    inputs=gpu(fixture['inputs']);mask=gpu(fixture['mask'])
    with torch.no_grad():
        cache=native_cache(model,inputs,gpu(fixture['noise']),gpu(fixture['t_v']),mask)
        flow=action_prediction(model,inputs,cache,gpu(fixture['noisy_action']),gpu(fixture['t_a']),mask)
        chunk=infer_from_cache(model,inputs,cache,mask,[fixture['position']],config['seed'])
    torch.testing.assert_close(flow.cpu(),fixture['expected_flow'],rtol=0,atol=0)
    torch.testing.assert_close(chunk.cpu(),fixture['expected_chunk'],rtol=0,atol=0)
    native_action=complete['provenance']['native_gpu_checks']['action_sha256']
    if tensor_hash(model.action_expert)!=native_action:raise RuntimeError('Export changed native Action2B')
    atomic_json(export/'RELOAD_CHECK.json',dict(passed=True,flow_exact=True,action_chunk_exact=True,
                 original_teacher_loaded=False,action_sha256=native_action,update=args.update))
    print('PASS: exported compact model reproduces flow and native ten-step actions exactly; Action2B unchanged',flush=True)

if __name__=='__main__':main()
