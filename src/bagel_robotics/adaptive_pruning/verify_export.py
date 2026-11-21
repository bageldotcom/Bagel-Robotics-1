"""Full-checkpoint structural comparison, separate from teacher-quality decisions."""
import argparse,json,os,gc
from pathlib import Path
import torch
from bagel_robotics._internal.adaptive_mask.geometry import Geometry
from bagel_robotics._internal.adaptive_mask.masking import install
from bagel_robotics._internal.adaptive_mask.export import build_compact, sliced_state, load_export
from bagel_robotics._internal.adaptive_mask.equivalence import streamed_fp32, compare, assess
from bagel_robotics._internal.adaptive_mask.data import prepare
from bagel_robotics._internal.compact_kv.model import load_reference, native_config, action_prediction
from bagel_robotics._internal.data.data import WindowReader
from bagel_robotics.data_selection.manifest import file_hash
from bagel_robotics._internal.compact_kv.spec import atomic_json
from bagel_robotics.adaptive_pruning.masks import Masks
from bagel_robotics.adaptive_pruning.config import load, identity
from bagel_robotics.adaptive_pruning.export import read_export_manifest


class BinaryAdapter:
    def __init__(self,selection,geometry,device):
        self.geometry=geometry;self.value=selection;self.device=device
    def selection(self):return self.value
    def forward_gates(self,mode='binary'):
        if mode!='binary':raise ValueError(mode)
        g=self.geometry;s=self.value
        h=torch.zeros(g.hidden,device=self.device);h[s['hidden_indices']]=1
        f=torch.zeros(g.layers-1,g.ffn,device=self.device);c=torch.zeros(g.layers-1,g.heads,device=self.device)
        for i in range(g.layers-1):f[i,s['ffn_indices'][i]]=1;c[i,s['cross_head_indices'][i]]=1
        return h,f,c


def run(config,initial=False,export_path=None):
    from omegaconf import OmegaConf
    from bagel_robotics.models.backbone.wan21_14b.wan_video_dit import WanVideoDiT, precompute_freqs_cis_3d
    from bagel_robotics.adaptive_pruning.preflight import require_receipt
    require_receipt(config,'cpu')
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('GPU verification requires Slurm')
    os.environ['VPP2_LOAD_META']='1'
    torch.cuda.set_device(0);device=torch.device('cuda',0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    g=Geometry(**config['geometry']);root=Path(config['root'])
    if initial:
        args=OmegaConf.to_container(native_config(config).model.video_dit_config,resolve=True)
        scores=torch.load(config['initial_masks'],map_location='cpu',weights_only=True)['state']['controller']
        selection=Masks(g,scores,config['initial_preferred']).selection()[0]
        trained=torch.load(Path(config['teacher_bundle'])/'video.pt',map_location='cpu',mmap=True,weights_only=True)['dit']
    else:
        export_path=Path(export_path or json.loads((root/'export_pending_verification.json').read_text())['path'])
        m=read_export_manifest(export_path)
        if m['identity']!=identity(config) or file_hash(export_path/'trained_wide.pt')!=m['verification_source_sha256']:
            raise ValueError('Selected trained student provenance changed')
        args=m['model_kwargs'];selection=m['selection']
        payload=torch.load(export_path/'trained_wide.pt',map_location='cpu',mmap=True,weights_only=True)
        if payload['checkpoint_identity']!=m['checkpoint_identity']:raise ValueError('Wrong trained checkpoint')
        trained=payload['dit']
    with torch.device('meta'):wide=WanVideoDiT(**args)
    wide.load_state_dict(trained,strict=True,assign=True);wide.freqs=precompute_freqs_cis_3d(g.head_dim)
    if initial:
        compact=build_compact(args,selection,g)
        state,_=sliced_state(trained,compact,selection,g);compact.load_state_dict(state,strict=True,assign=True)
    else:compact=load_export(export_path)
    wide=wide.bfloat16().requires_grad_(False);compact=compact.bfloat16().requires_grad_(False)
    wb=install(wide).eval();cb=install(compact).eval()
    controller=BinaryAdapter(selection,g,device)
    reference=load_reference(config,device='cpu');reference.device=device
    for name in ('action_expert','proprio_encoder','clip_encoder','vae','text_encoder'):
        module=getattr(reference,name,None)
        if module is not None:module.to(device)
    reference._restore_auxiliary_component_dtypes();reference.action_expert.use_gradient_checkpointing=False;reference.eval()
    reader=WindowReader(config);reference._full_mask_conditioning_cache=reader.runtime_cache
    panel=json.loads(Path(config['panel_file']).read_text())
    rows=[]
    for suite in ('memory','precision'):
        rows.append(next(r for r in panel['windows'] if r['role']=='selection_initial' and r['suite']==suite))
    checks=[]
    with torch.no_grad():
        for row in rows:
            parts=prepare(reference,reader,[row],[-4_000_000-row['id']],config['seed'],[2])
            if initial:
                # Compare against the untouched native implementation, before using
                # any masking wrapper as the reference for physical surgery.
                reference.video_expert.to(device)
                inp=parts[0]
                nc,nm,nl,nv=reference._build_frozen_video_cache(latents=parts[1],timestep_video=parts[2],
                    context=inp['context'],context_mask=inp['context_mask'],condition_latents=inp['condition_latents'],
                    clip_fea=inp['clip_fea'],action_seq_len=32,fuse_vae_embedding_in_latents=False,return_video_prediction=True)
                expected_video=nv.cpu();expected_cache=[{k:v.cpu() for k,v in x.items()} for x in nc]
                expected_action=action_prediction(reference,inp,nc,parts[3],parts[4],parts[6]).cpu()
                del nc,nv;reference.video_expert.cpu();torch.cuda.empty_cache()
                wb.to(device);opened=wb(parts[0],parts[1],parts[2],parts[6][:1950,:1950])
                torch.testing.assert_close(opened['video'].cpu(),expected_video,rtol=0,atol=0)
                for x,y in zip(opened['cache'],expected_cache):
                    for key in ('k','v'):torch.testing.assert_close(x[key].cpu(),y[key],rtol=0,atol=0)
                torch.testing.assert_close(action_prediction(reference,inp,opened['cache'],parts[3],parts[4],parts[6]).cpu(),expected_action,rtol=0,atol=0)
                del opened,expected_cache,expected_video,expected_action;wb.cpu();torch.cuda.empty_cache()
            fp=streamed_fp32(wide,compact,parts,controller,progress=lambda x:print(json.dumps(x),flush=True))
            wb.to(device);a=wb(parts[0],parts[1],parts[2],parts[6][:1950,:1950],*controller.forward_gates())
            aa=action_prediction(reference,parts[0],a['cache'],parts[3],parts[4],parts[6]).cpu()
            cached={k:(v.cpu() if torch.is_tensor(v) else [{kk:vv.cpu() for kk,vv in entry.items()} for entry in v]) for k,v in a.items()}
            del a;wb.cpu();torch.cuda.empty_cache()
            cb.to(device);b=cb(parts[0],parts[1],parts[2],parts[6][:1950,:1950])
            ba=action_prediction(reference,parts[0],b['cache'],parts[3],parts[4],parts[6]).cpu()
            bf=[compare(b['video'].cpu(),cached['video'],'video'),compare(ba,aa,'action')]
            bf += [compare(right[key].cpu(),left[key],f'layer_{i:02d}_{key}') for i,(left,right) in enumerate(zip(cached['cache'],b['cache'])) for key in ('k','v')]
            # Record compact outputs for ordinary export reload parity.
            bv=b['video'].cpu();bc=[{k:v.cpu() for k,v in entry.items()} for entry in b['cache']]
            del b;cb.cpu();torch.cuda.empty_cache()
            if not initial:
                from bagel_robotics.adaptive_pruning.monitor import reader_manifest
                from bagel_robotics.adaptive_pruning.monitor import NativeInferenceReader,masked_prefill
                native_inputs,_=NativeInferenceReader(config,reader_manifest([row])).read(row['episode'],row['frame'])
                cb.to(device)
                def prefill(inputs,noise,timestep,mask,*unused):
                    return cb(inputs,noise,timestep,mask,return_video=False)
                with masked_prefill(reference,prefill):generated=reference.infer_action(**native_inputs)['action'].cpu()
                cb.cpu();torch.cuda.empty_cache()
                local_config=dict(config,teacher_bundle=str(export_path/'native'))
                reloaded=load_reference(local_config,device='cpu');reloaded.device=device
                reloaded.video_expert.to(device);reloaded.action_expert.to(device)
                reloaded.proprio_encoder.to(device)
                # Identical frozen encoders can be reused. No teacher video or masks
                # are attached to the reloaded, ordinary native inference policy.
                reloaded.vae=reference.vae;reloaded.clip_encoder=reference.clip_encoder
                reloaded.action_expert.use_gradient_checkpointing=False
                inp=parts[0]
                rc,rm,rl,rv=reloaded._build_frozen_video_cache(latents=parts[1],timestep_video=parts[2],
                    context=inp['context'],context_mask=inp['context_mask'],condition_latents=inp['condition_latents'],
                    clip_fea=inp['clip_fea'],action_seq_len=32,fuse_vae_embedding_in_latents=False,return_video_prediction=True)
                ra=action_prediction(reloaded,inp,rc,parts[3],parts[4],parts[6]).cpu()
                torch.testing.assert_close(ra,ba,rtol=0,atol=0)
                torch.testing.assert_close(rv.cpu(),bv,rtol=0,atol=0)
                for left,right in zip(bc,rc):
                    for key in ('k','v'):torch.testing.assert_close(left[key],right[key].cpu(),rtol=0,atol=0)
                native_generated=reloaded.infer_action(**native_inputs)['action'].cpu()
                torch.testing.assert_close(native_generated,generated,rtol=0,atol=0)
                del reloaded,rc,rv;gc.collect();torch.cuda.empty_cache()
            summary=assess(fp,bf,40,2e-5,.01)
            checks.append(dict(window_id=row['id'],suite=row['suite'],summary=summary,fp32=fp,bf16=bf))
    report=dict(identity=identity(config),passed=all(x['summary']['passed'] for x in checks),checks=checks,
                policy_retention_established=False,teacher_closeness_gate=False,initial_selection_only=initial,
                original_native_all_open_verified=initial,native_no_adapter_reload_verified=not initial)
    if initial:atomic_json(root/'checks/structure.json',report)
    else:
        report['manifest_sha256']=file_hash(export_path/'manifest.json')
        atomic_json(export_path/'VERIFIED.json',report)
        if report['passed']:atomic_json(root/'export_verified.json',dict(identity=identity(config),path=str(export_path),passed=True))
    if not report['passed']:raise RuntimeError('Physical slicing equivalence failed; inspect comparison report')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--initial',action='store_true');p.add_argument('--export')
    a=p.parse_args();run(load(a.config),a.initial,a.export)

if __name__=='__main__':main()
