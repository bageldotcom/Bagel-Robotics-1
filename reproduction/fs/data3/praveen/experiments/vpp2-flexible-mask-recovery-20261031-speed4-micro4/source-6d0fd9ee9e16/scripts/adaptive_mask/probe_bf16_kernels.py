"""Isolate real checkpoint GEMMs; diagnostics only, no training or weight mutation."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F

p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
c=json.loads(Path(a.config).read_text());torch.manual_seed(100008)
state=torch.load(Path(c['teacher_bundle'])/'video.pt',map_location='cpu',mmap=True,weights_only=True)['dit']
hidden=torch.randperm(5120)[:4864].sort().values.cuda()
rows=[]
for name,inp,out in [('blocks.0.self_attn.k',hidden,None),('blocks.0.ffn.0',hidden,torch.arange(13440,device='cuda')),
                     ('blocks.0.ffn.2',torch.arange(13440,device='cuda'),hidden),('time_projection.1',hidden,None)]:
    w=state[name+'.weight'].cuda().bfloat16();b=state[name+'.bias'].cuda().bfloat16()
    x=torch.randn(1,1950,w.shape[1],device='cuda',dtype=torch.bfloat16)
    gate=torch.zeros(w.shape[1],device='cuda',dtype=torch.bfloat16);gate[inp]=1;x=x*gate
    kept_w=w.index_select(1,inp);kept_b=b
    if out is not None:kept_w=kept_w.index_select(0,out);kept_b=kept_b.index_select(0,out)
    for reduced in [True,False]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=reduced
        wide=F.linear(x,w,b);slim=F.linear(x.index_select(-1,inp),kept_w,kept_b)
        if out is not None:wide=wide.index_select(-1,out)
        error=(wide.float()-slim.float())
        rows.append(dict(module=name,reduced_precision_reduction=reduced,
                         relative_rmse=float(error.square().mean().sqrt()/wide.float().square().mean().sqrt()),
                         differing_fraction=float((wide!=slim).float().mean()),max_absolute=float(error.abs().max())))
    torch.backends.cuda.matmul.allow_tf32=False
    wide=F.linear(x.float(),w.float(),b.float());slim=F.linear(x.index_select(-1,inp).float(),kept_w.float(),kept_b.float())
    if out is not None:wide=wide.index_select(-1,out)
    error=wide-slim
    rows.append(dict(module=name,precision='FP32',relative_rmse=float(error.square().mean().sqrt()/wide.square().mean().sqrt()),
                     max_absolute=float(error.abs().max())))
    del x,w,b,kept_w,kept_b,wide,slim,error
    torch.cuda.empty_cache()
output=Path(c['root'])/'checks/bf16_kernel_diagnostic.json';output.write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows),flush=True)
