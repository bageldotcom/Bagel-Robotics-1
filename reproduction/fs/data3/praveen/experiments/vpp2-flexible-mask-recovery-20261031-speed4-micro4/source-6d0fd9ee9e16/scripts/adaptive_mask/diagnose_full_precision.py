"""Full-checkpoint, matched-input precision diagnosis; never a readiness receipt."""
import argparse,json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from vpp2.adaptive_mask.runtime import initialize,forward_video,predict
from vpp2.adaptive_mask.data import prepare
from vpp2.adaptive_mask.controller import ranking
from vpp2.adaptive_mask.export import build_compact,sliced_state
from vpp2.adaptive_mask.masking import install
from vpp2.adaptive_mask.preflight import error_report
from vpp2.compact_kv.model import native_config


def compare(a,b,name):return dict(tensor=name,**error_report(a,b))


@torch.no_grad()
def streamed_fp32(wide,compact,parts,controller):
    # Only the current pair of blocks occupies GPU RAM. Original checkpoint
    # values are BF16; conversion to FP32 is exact and neither model is trained.
    for model in (wide,compact):
        for name,child in model.named_children():
            if name!='blocks':child.to(device='cuda',dtype=torch.float32)
    inputs={k:v.float() if torch.is_tensor(v) and v.is_floating_point() else v for k,v in parts[0].items()}
    noise=parts[1].float();t=parts[2].float();mask=parts[6][:1950,:1950]
    gates=controller.forward_gates(mode='binary');indices=torch.tensor(controller.selection()['hidden_indices'],device='cuda')
    a=wide.pre_dit(noise,t,inputs['context'],inputs['context_mask'],inputs['clip_fea'],inputs['condition_latents'],hidden_gate=gates[0])
    b=compact.pre_dit(noise,t,inputs['context'],inputs['context_mask'],inputs['clip_fea'],inputs['condition_latents'])
    result=[compare(a[k].index_select(-1,indices),b[k],'embedding_'+k) for k in ('tokens','t','t_mod','context')]
    x,y=a['tokens'],b['tokens']
    for i,(left,right) in enumerate(zip(wide.blocks,compact.blocks)):
        left.to(device='cuda',dtype=torch.float32);right.to(device='cuda',dtype=torch.float32)
        f=None if i==39 else gates[1][i];h=None if i==39 else gates[2][i]
        x,k,v=left(x,a['t_mod'],a['freqs'],a['context'],a['context_mask'],mask,gates[0],f,h)
        y,kk,vv=right(y,b['t_mod'],b['freqs'],b['context'],b['context_mask'],mask)
        result.extend([compare(k,kk,f'layer_{i:02d}_k'),compare(v,vv,f'layer_{i:02d}_v'),
                       compare(x.index_select(-1,indices),y,f'layer_{i:02d}_hidden')])
        left.to(device='cpu',dtype=torch.bfloat16);right.to(device='cpu',dtype=torch.bfloat16)
        del k,v,kk,vv
        if i%10==0:print(json.dumps(dict(event='fp32_layer',layer=i,worst=max(r['relative_rmse'] for r in result))),flush=True)
    av=wide.unpatchify(wide.head(x,a['t'],gates[0]),a['meta']['grid_size'])
    bv=compact.unpatchify(compact.head(y,b['t']),b['meta']['grid_size'])
    result.append(compare(av,bv,'video'))
    return result


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    c=json.loads(Path(args.config).read_text());panel=json.loads(Path(c['panel_file']).read_text())
    model,reader,controller,branch,inventory=initialize(c)
    row=next(r for r in panel['windows'] if r['role']=='selection_initial' and r['suite']=='precision')
    parts=prepare(model,reader,[row],[12345],c['seed'],[2])
    kwargs=OmegaConf.to_container(native_config(c).model.video_dit_config,resolve=True)
    reports={};output=Path(c['root'])/'checks/full_precision_diagnostic.json'
    for width in [5120,4992,4864]:
        selection=controller.geometry.full()
        selection['hidden_indices']=sorted(ranking(controller.hidden_scores)[:width])
        selection['ffn_indices']=[list(range(13440)) for _ in range(39)]+[list(range(13824))]
        selection['cross_head_indices']=[list(range(39)) for _ in range(39)]+[list(range(40))]
        controller.set_selection(selection);model.video_expert.cuda();torch.cuda.empty_cache()
        wide=forward_video(branch,parts,controller.forward_gates(mode='binary'))
        wa=predict(model,parts,wide['cache']).cpu()
        wc=[{k:v.cpu() for k,v in layer.items()} for layer in wide['cache']];wv=wide['video'].cpu()
        del wide
        model.video_expert.cpu();torch.cuda.empty_cache()
        compact=build_compact(kwargs,selection,controller.geometry)
        state,_=sliced_state(model.video_expert.state_dict(),compact,selection,controller.geometry)
        compact.load_state_dict(state,strict=True,assign=True);del state
        compact=compact.to(device='cuda',dtype=torch.bfloat16);cb=install(compact).eval()
        slim=forward_video(cb,parts);sa=predict(model,parts,slim['cache']).cpu()
        values=[compare(layer[k].cpu(),ref[k],f'layer_{i:02d}_{k}') for i,(layer,ref) in enumerate(zip(slim['cache'],wc)) for k in ('k','v')]
        values.extend([compare(slim['video'].cpu(),wv,'video'),compare(sa,wa,'action')])
        reports[str(width)]=dict(bf16=values,bf16_max=max(r['relative_rmse'] for r in values))
        output.write_text(json.dumps(dict(diagnostic_only=True,results=reports),indent=2)+'\n')
        print(json.dumps(dict(event='bf16_width',width=width,maximum=reports[str(width)]['bf16_max'],actions=values[-1])),flush=True)
        del slim,wc,wv,wa,sa
        compact.cpu();torch.cuda.empty_cache()
        if width==4864:
            torch.backends.cuda.matmul.allow_tf32=False
            reports[str(width)]['fp32']=streamed_fp32(model.video_expert,compact,parts,controller)
            reports[str(width)]['fp32_max']=max(r['relative_rmse'] for r in reports[str(width)]['fp32'])
            output.write_text(json.dumps(dict(diagnostic_only=True,results=reports),indent=2)+'\n')
        del cb,compact
    print(json.dumps(dict(event='complete',summary={w:{k:v for k,v in r.items() if k.endswith('_max')} for w,r in reports.items()})),flush=True)


if __name__=='__main__':main()
