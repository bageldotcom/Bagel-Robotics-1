"""Fail-closed CPU, native GPU, full-batch distributed and resume validation."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from vpp2.full_mask.panel import digest,file_hash,immutable,BalancedStream,noise_stratum
from .state import identity,rng_state,restore_rng
from .runtime import distributed,initialize,forward_video,predict,check_frozen,model_versions,gather,barrier,step
from .data import prepare,raw_reader
from .geometry import Geometry,ParameterInventory
from .controller import Controller


def validate_inputs(config,panel):
    from .quality_policy import policy_mode
    policy_mode(config)
    if digest({k:v for k,v in panel.items() if k!='identity'})!=panel['identity']:raise ValueError('Panel modified')
    if file_hash(config['panel_file'])!=config['artifacts']['panel_sha256']:raise ValueError('Panel bytes changed')
    if len([r for r in panel['windows'] if r['role']=='selection_initial'])!=30000:raise ValueError('Expected 30k fitting windows')
    if len(panel['tasks'])!=35:raise ValueError('Task scope changed')
    roles={}
    for row in panel['windows']:
        if row['role'] not in ('selection_initial','monitor'):continue
        old=roles.setdefault(row['episode'],row['role'])
        if old!=row['role'] or row['episode'] in panel['official_validation_episodes']:raise ValueError('Episode leakage')
    for key in ('normalization','source_manifest','visual_review','data_cpu_receipt','audit_manifest'):
        artifact=config['artifacts'][key]
        if file_hash(artifact['path'])!=artifact['sha256']:raise ValueError(key+' identity changed')
    data_check=json.loads(Path(config['artifacts']['data_cpu_receipt']['path']).read_text())
    if not data_check.get('passed') or data_check.get('panel_identity')!=panel['identity']:raise ValueError('Original data integrity check does not cover panel')
    stats=json.loads(Path(config['artifacts']['normalization']['path']).read_text())
    if stats!=json.loads((Path(config['teacher_bundle'])/'dataset_stats.json').read_text()):raise ValueError('Normalization mismatch')
    visual=json.loads(Path(config['artifacts']['visual_review']['path']).read_text())
    if visual.get('passed') is not True or visual.get('panel_identity')!=panel['identity']:raise ValueError('Unverified visual review')
    if config['train']['action_gt_weight']!=1 or config['train']['teacher_action_weight']!=0:raise ValueError('Action objective changed')
    if config['train']['kv_layers']!=list(range(20,40)):raise ValueError('Late cache layer set changed')
    if config['train']['kv_weight']!=.05 or config['train']['teacher_video_weight']!=.05:raise ValueError('Teacher coefficients changed')
    if config['budget']['protect_hidden']:raise ValueError('This revision requires learnable hidden scores')
    if config['hidden_search']['widths']!=[5120,4992,4864] or config['geometry']['hidden_floor']!=4864:
        raise ValueError('Hidden search must be bounded to 0–5 percent')
    if not config['train']['frozen_video'] or not config['train']['frozen_action']:raise ValueError('Only mask scores may train')
    if config['hidden_search']['trial_updates']%config['train']['monitor_every']:raise ValueError('Hidden trial check cadence mismatch')
    if config['hidden_search']['last_trial_update']+config['hidden_search']['trial_updates']>config['budget']['milestones'][-1][0]:
        raise ValueError('Hidden trial extends into final settling period')
    if config['train']['max_full_checks_excluding_initial']!=8:raise ValueError('Hidden trial verification budget changed')
    if config['reliability']['calibration_passes']!=0 or config['reliability']['method']!='clipped_relative_gt_flow_error':raise ValueError('Reliability recipe changed')
    if config['train']['budget_weight'] or config['train']['entropy_weight'] or config['train']['discreteness_weight']:raise ValueError('Unexpected soft-budget penalty')
    from .equivalence import POLICY
    if config['validation'].get('export_equivalence_policy') != POLICY:
        raise ValueError('Explicit approved export equivalence policy required')
    if config['validation']['fp32_relative_tolerance'] != 2e-5 or config['validation']['binary_export_relative_rmse_max'] != .01:
        raise ValueError('Approved structural/output tolerances changed')
    from .audit import validate_manifest
    validate_manifest(json.loads(Path(config['artifacts']['audit_manifest']['path']).read_text()),panel)
    feedback=config['capacity_feedback']
    if feedback['source']!='fitting_only' or feedback['reliability_filtered'] or feedback['every']!=config['train']['monitor_every']:
        raise ValueError('Invalid capacity feedback data role or cadence')
    return dict(fitting_windows=30000,monitor_windows=sum(r['role']=='monitor' for r in panel['windows']),
        tasks=35,fitting_episodes=sum(role=='selection_initial' for role in roles.values()))


def cpu(config,panel):
    scope=validate_inputs(config,panel);root=Path(config['root']);ident=identity(config,panel)
    source=Path(__file__).resolve().parents[3]
    result=subprocess.run([sys.executable,'-m','pytest','-q',str(source/'tests/adaptive_mask'),
                           '--junitxml='+str(root/'checks/unit_tests.xml')],check=False)
    if result.returncode:raise RuntimeError('CPU tests failed; no readiness receipt')
    g=Geometry(**config['geometry']);teacher=torch.load(Path(config['teacher_bundle'])/'video.pt',mmap=True,map_location='cpu',weights_only=True)['dit']
    inv=ParameterInventory({k:v.shape for k,v in teacher.items()},g)
    if inv.original!=config['original_video_parameters']:raise ValueError('Native tensor count differs')
    c=Controller(g,config['seed']);audits=[]
    for h,cap in [(h,cap) for h in config['hidden_search']['widths'] for cap in (inv.original,10_000_000_000)]:
        allocation=c.project(inv,cap,h);actual=inv.count(c.selection())
        if actual!=allocation['parameters'] or actual>cap:raise AssertionError('Exact budget audit failed')
        audits.append(dict(**allocation,exact_tensor_count=actual))
    immutable(root/'checks/cpu.json',dict(passed=True,identity=ident,scope=scope,physical_budget_audits=audits,
        calibration_passes=0,extra_calibration_action_predictions=0,native_gpu_checks_performed=False))



def error_report(a,b):
    diff=(a.float()-b.float());scale=b.float().square().mean().sqrt().clamp_min(1e-6)
    return dict(max_absolute=float(diff.abs().max()),relative_rmse=float(diff.square().mean().sqrt()/scale))


def native(config,panel,final=False):
    rank,world=distributed()
    if world!=1:raise ValueError('Native check requires one allocated GPU')
    ident=identity(config,panel);root=Path(config['root'])
    receipt=json.loads((root/'checks/cpu.json').read_text())
    if receipt.get('identity')!=ident or not receipt.get('passed'):raise ValueError('CPU preflight stale')
    from vpp2.critical_windows.inference import model_identity
    from vpp2.compact_kv.gpu_check import tensor_hash
    from .masking import install
    from .monitor import native_evaluate
    from .losses import coordinate_max_error
    from vpp2.compact_kv.losses import action_error
    model,reader,controller,branch,inventory=initialize(config,install_branch=False)
    before_video=tensor_hash(model.video_expert);before_action=tensor_hash(model.action_expert);versions=model_versions(model)
    row=next(r for r in panel['windows'] if r['role']=='selection_initial' and r['suite']=='precision')
    parts=prepare(model,reader,[row],[12345],config['seed'],[2]);inp=parts[0]
    with torch.no_grad():
        caches,mask,length,video=model._build_frozen_video_cache(latents=parts[1],timestep_video=parts[2],
            context=inp['context'],context_mask=inp['context_mask'],condition_latents=inp['condition_latents'],clip_fea=inp['clip_fea'],
            action_seq_len=32,fuse_vae_embedding_in_latents=False,return_video_prediction=True)
        action=predict(model,parts,caches)
    branch=install(model.video_expert);branch.train()
    with torch.no_grad():
        opened=forward_video(branch,parts);hard_open=forward_video(branch,parts,controller.forward_gates(mode='binary'))
        for out in (opened,hard_open):
            if len(out['cache'])!=40 or out['video'].shape!=(1,16,5,30,52):raise AssertionError('Wrong native dimensions')
            if not torch.equal(video,out['video']):raise AssertionError('All-open video differs')
            for a,b in zip(caches,out['cache']):
                for k in ('k','v'):
                    if a[k].shape!=(1,1950,5120) or not torch.equal(a[k],b[k]):raise AssertionError('All-open native cache differs')
            if not torch.equal(action,predict(model,parts,out['cache'])):raise AssertionError('All-open action differs')
    del opened,hard_open,caches,video,action
    # Explicit future-label causality: modify future RGB while leaving history fixed.
    from vpp2.full_mask.data import collate,prepared
    with raw_reader(reader,future=True):sample=collate(reader,[row],[12345])
    cache_pointer=model._full_mask_conditioning_cache;model._full_mask_conditioning_cache=None
    a=prepared(model,sample,config['seed'],[2]);sample['video'][:,:,9:]=1-sample['video'][:,:,9:]
    b=prepared(model,sample,config['seed'],[2]);model._full_mask_conditioning_cache=cache_pointer
    with torch.no_grad():
        left=forward_video(branch,a);right=forward_video(branch,b)
        if not torch.equal(left['video'],right['video']):raise AssertionError('Future labels change prediction')
        if not torch.equal(predict(model,a,left['cache']),predict(model,b,right['cache'])):raise AssertionError('Future-label action leakage')
    del a,b,left,right,sample
    # Test nontrivial hidden/FFN/head masks, including regrowth through GT actions.
    selection=controller.geometry.full()
    from .controller import ranking
    selection['hidden_indices']=sorted(ranking(controller.hidden_scores)[:config['hidden_search']['widths'][-1]])
    selection['ffn_indices']=[list(range(13440)) for _ in range(39)]+[list(range(13824))]
    selection['cross_head_indices']=[list(range(39)) for _ in range(39)]+[list(range(40))]
    if final:
        final_record=json.loads((root/'final_selection.json').read_text())
        if not final_record.get('stable') or final_record['identity']!=ident:raise ValueError('Final selection is not validated/stable')
        selection=final_record['selection']
    controller.set_selection(selection)
    output=forward_video(branch,parts,controller.forward_gates());prediction=predict(model,parts,output['cache'])
    loss=action_error(prediction,parts[5],inp['action_loss_is_pad']).mean()
    loss.backward()
    gradients={name:float(p.grad.norm()) if p.grad is not None else 0. for name,p in controller.named_parameters() if p.requires_grad}
    if not all(v>0 for v in gradients.values()) or not all(torch.isfinite(p.grad).all() for p in controller.trainable_parameters()):raise AssertionError('Missing/nonfinite mask gradients')
    for scores,gate in zip(controller.parameters(),controller.gates(mode='binary')):
        if scores.requires_grad and (gate==0).any() and not scores.grad[gate==0].abs().sum()>0:raise AssertionError('Removed units cannot receive regrowth gradients')
    if controller.hidden_scores.grad is None or controller.hidden_scores.grad.norm()==0:raise AssertionError('No GT action signal reaches hidden scores')
    opt=torch.optim.AdamW(controller.trainable_parameters(),lr=.01,weight_decay=0.)
    from .priority import MaskPriority
    priority=MaskPriority(controller,**config['mask_priority'])
    for _ in range(config['mask_priority']['checks']):priority.observe()
    priority.step(opt);opt.zero_grad(set_to_none=True)
    check_frozen(model,versions)
    if before_video!=tensor_hash(model.video_expert) or before_action!=tensor_hash(model.action_expert):raise AssertionError('Frozen weights changed')
    del output,prediction,loss
    # Store binary reference before swapping the large video network out of GPU RAM.
    with torch.no_grad():
        binary=forward_video(branch,parts,controller.forward_gates(mode='binary'))
        expected_cache=[{k:v.cpu() for k,v in c.items()} for c in binary['cache']]
        expected_video=binary['video'].cpu();expected_action=predict(model,parts,binary['cache']).cpu()
    del binary
    from .export import build_compact,sliced_state
    from vpp2.compact_kv.model import native_config
    from omegaconf import OmegaConf
    args=OmegaConf.to_container(native_config(config).model.video_dit_config,resolve=True)
    model.video_expert.cpu();torch.cuda.empty_cache()
    compact=build_compact(args,selection,controller.geometry)
    sliced,audit=sliced_state(model.video_expert.state_dict(),compact,selection,controller.geometry)
    compact.load_state_dict(sliced,strict=True,assign=True);del sliced
    compact=compact.to(device=parts[1].device,dtype=torch.bfloat16)
    compact_branch=install(compact).eval()
    with torch.no_grad():
        actual=forward_video(compact_branch,parts);actual_action=predict(model,parts,actual['cache'])
        comparisons=[dict(tensor=f'layer_{i:02d}_{k}',**error_report(c[k].cpu(),e[k]))
                     for i,(c,e) in enumerate(zip(actual['cache'],expected_cache)) for k in ('k','v')]
        comparisons += [dict(tensor='video',**error_report(actual['video'].cpu(),expected_video)),
                        dict(tensor='action',**error_report(actual_action.cpu(),expected_action))]
    tolerance=config['validation']['binary_export_relative_rmse_max']
    maximum=max(c['relative_rmse'] for c in comparisons)
    del actual,actual_action,expected_cache,expected_video,expected_action
    compact.cpu();torch.cuda.empty_cache()
    from .equivalence import streamed_fp32,assess
    print(json.dumps(dict(event='native_fp32_start',bf16_maximum_relative_rmse=maximum)),flush=True)
    fp32_comparisons=streamed_fp32(model.video_expert,compact,parts,controller,
        progress=lambda value:print(json.dumps(dict(event='native_fp32_layer',**value)),flush=True))
    equivalence=assess(fp32_comparisons,comparisons,controller.geometry.layers,
                      config['validation']['fp32_relative_tolerance'],tolerance)
    # Preserve every BF16 cache difference. The approved check requires full FP32
    # structural equivalence and <1% BF16 action-flow/video output differences.
    immutable(root/f"checks/{'final' if final else 'native'}_comparison_{os.environ.get('SLURM_JOB_ID','local')}.json",
        dict(identity=ident,comparisons=comparisons,fp32_comparisons=fp32_comparisons,
             maximum_relative_rmse=maximum,tolerance=tolerance,equivalence=equivalence,
             comparison_passed=equivalence['passed'],all_open_bitwise=True,future_target_invariance=True,
             gradient_objective='ground_truth_actions_only',mask_gradient_norms=gradients,hidden_width=len(selection['hidden_indices'])))
    if not equivalence['passed']:raise AssertionError('Binary/export checks failed: '+str(equivalence))
    if before_video!=tensor_hash(model.video_expert) or before_action!=tensor_hash(model.action_expert):
        raise AssertionError('Precision verification changed frozen checkpoint values')
    if sum(p.numel() for p in compact.parameters())!=inventory.count(selection):raise AssertionError('Physical parameter inventory differs')
    if final:
        from .export import publish
        publish(config,selection,controller.geometry,args,ident)
    immutable(root/('checks/final_export.json' if final else 'checks/native.json'),dict(passed=True,identity=ident,checkpoint_hashes=model_identity(config),
        all_open_cache_action_video_bitwise=True,future_target_invariance=True,frozen_hashes_unchanged=True,
        mask_gradient_norms=gradients,gradient_objective='ground_truth_actions_only',removed_channel_gradients=True,hidden_width=len(selection['hidden_indices']),binary_export_comparisons=comparisons,
        binary_export_tolerance=tolerance,export_equivalence=equivalence,fp32_comparisons=fp32_comparisons,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        selected_physical_parameters=sum(p.numel() for p in compact.parameters()),simulation=False))


def distributed_check(config,panel):
    rank,world=distributed();root=Path(config['root']);ident=identity(config,panel)
    if world!=8:raise ValueError('Requires 8 Slurm-allocated GPUs')
    checked=json.loads((root/'checks/native.json').read_text())
    if not checked['passed'] or checked['identity']!=ident:raise ValueError('Native GPU check stale')
    from .state import snapshot,restore
    from vpp2.full_mask.distributed_smoke import assert_equal
    from .controller import synchronize_controller
    model,reader,controller,branch,inventory=initialize(config);versions=model_versions(model)
    # Verify native execution on every training task, including low-motion tasks.
    rows=[next(r for r in panel['windows'] if r['role']=='selection_initial' and r['task']==task) for task in panel['tasks']]
    coverage=[]
    controller.project(inventory,inventory.full_at_hidden(config['hidden_search']['widths'][-1])-250_000_000,config['hidden_search']['widths'][-1])
    for i,row in enumerate(rows):
        if i%world!=rank:continue
        parts=prepare(model,reader,[row],[-4_000_000-row['id']],config['seed'],[2])
        loss,metrics,details,_=step(model,branch,controller,parts,[row],100,config)
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in controller.trainable_parameters()):raise AssertionError('Task execution failed')
        controller.zero_grad(set_to_none=True);coverage.append(row['task']);del loss,metrics,details,parts
    coverage=sum(gather(coverage),[])
    if sorted(coverage)!=sorted(panel['tasks']):raise AssertionError('Task probe coverage incomplete')
    stream=BalancedStream([r for r in panel['windows'] if r['role']=='selection_initial'],config['seed'])
    optimizer=torch.optim.AdamW(controller.trainable_parameters(),lr=.01,betas=(.9,.999),weight_decay=0.)
    from .priority import MaskPriority
    priority=MaskPriority(controller,**config['mask_priority'])
    for _ in range(config['mask_priority']['checks']):priority.observe()
    performance=[]
    def update(index):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        optimizer.zero_grad(set_to_none=True);total=0.
        width=config['hidden_search']['widths'][1 if index==0 else 2]
        controller.project(inventory,inventory.full_at_hidden(width)-250_000_000,width)
        for micro in range(32):
            position=index*256+micro*8+rank;row=stream.at(position);s=noise_stratum(config['seed'],position,35)
            parts=prepare(model,reader,[row],[position],config['seed'],[s])
            loss,metrics,details,_=step(model,branch,controller,parts,[row],100,config)
            (loss/32).backward();total=total+loss.detach()/32
            del loss,metrics,details,parts
        synchronize_controller(controller,1.);priority.step(optimizer);priority.refresh();check_frozen(model,versions)
        torch.cuda.synchronize()
        measured=gather(dict(rank=rank,seconds=time.perf_counter()-started,
                             peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                             peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        performance.append(dict(update=index,ranks=measured,slowest_seconds=max(r['seconds'] for r in measured)))
        if rank==0:print(json.dumps(dict(event='distributed_throughput',**performance[-1])),flush=True)
        return total.cpu()
    update(0);start=copy.deepcopy(controller.state_dict());opt_start=copy.deepcopy(optimizer.state_dict());priority_start=priority.state_dict();rng=rng_state()
    expected_loss=update(1);expected=copy.deepcopy(controller.state_dict());opt_expected=copy.deepcopy(optimizer.state_dict());priority_expected=priority.state_dict()
    controller.load_state_dict(start);optimizer.load_state_dict(opt_start);priority.load_state_dict(priority_start);restore_rng(rng)
    actual_loss=update(1);assert_equal(expected,controller.state_dict());assert_equal(opt_expected,optimizer.state_dict());assert_equal(expected_loss,actual_loss);assert_equal(priority_expected,priority.state_dict())
    for parameter in controller.parameters():
        ref=parameter.detach().clone();torch.distributed.broadcast(ref,src=0)
        if not torch.equal(ref,parameter):raise AssertionError('Rank mask divergence')
    if rank==0:immutable(root/'checks/distributed.json',dict(passed=True,identity=ident,world_size=8,global_batch=256,micro_batch=1,
        execution=config.get('execution',{}),performance=performance,
        task_probe_coverage=coverage,hidden_widths_tested=config['hidden_search']['widths'][1:],resume_bitwise_equal=True,frozen_weights_unchanged=True,fresh_main_optimizer=True,main_cursor=0,
        note='Smoke optimizer/mask updates discarded; main training starts from fresh native weights and mask scores.'))
    barrier()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--mode',choices=['cpu','native','distributed','final-export'],required=True);a=p.parse_args()
    c=json.loads(Path(a.config).read_text());panel=json.loads(Path(c['panel_file']).read_text())
    validate_inputs(c,panel);Path(c['root'],'checks').mkdir(parents=True,exist_ok=True)
    if a.mode=='final-export':
        if not (Path(c['root'])/'final_selection.json').exists():
            print('No stable final selection: physical export deliberately not produced.');return
        audit=json.loads((Path(c['root'])/'audit/report.json').read_text())
        if not audit.get('evaluated') or audit['identity']!=identity(c,panel) or audit['final_selection_sha256']!=file_hash(Path(c['root'])/'final_selection.json'):
            raise ValueError('Completed audit does not match the exported selection')
        native(c,panel,final=True)
    else:{'cpu':cpu,'native':native,'distributed':distributed_check}[a.mode](c,panel)

if __name__=='__main__':main()
