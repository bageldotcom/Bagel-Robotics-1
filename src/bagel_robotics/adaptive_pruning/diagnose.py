"""Fixed native inference and flow diagnostics, isolated from training state."""
from contextlib import contextmanager
import torch
import torch.distributed as dist
from bagel_robotics._internal.training.rng import diagnostic_rng
from bagel_robotics._internal.adaptive_mask.losses import component_errors
from bagel_robotics.adaptive_pruning.monitor import select_windows, reader_manifest
from bagel_robotics.adaptive_pruning.monitor import NativeInferenceReader, masked_prefill
from bagel_robotics.adaptive_pruning.selection import aggregate


@contextmanager
def diagnostic_mode(runtime):
    modules=list(runtime.reference.modules())+list(runtime.engine.modules())+list(runtime.teacher.modules())
    old=[m.training for m in modules]
    action_flag=runtime.reference.action_expert.use_gradient_checkpointing
    try:
        runtime.engine.eval();runtime.teacher.eval();runtime.reference.eval()
        runtime.reference.action_expert.use_gradient_checkpointing=False
        with diagnostic_rng(),torch.no_grad():yield
    finally:
        for m,flag in zip(modules,old):m.training=flag
        runtime.reference.action_expert.use_gradient_checkpointing=action_flag


def padded_rows(rows,rank,world):
    # ZeRO-3 forward collectives need an equal number of calls on every rank.
    for start in range(0,len(rows),world):
        index=start+rank
        yield rows[index if index<len(rows) else 0],index<len(rows)


def gather_rows(rows,expected):
    parts=[None]*dist.get_world_size();dist.all_gather_object(parts,rows)
    result=sorted(sum(parts,[]),key=lambda r:r['window_id'])
    if len(result)!=len(expected) or {r['window_id'] for r in result}!={r['id'] for r in expected}:
        raise RuntimeError('Incomplete or duplicated distributed diagnostic')
    return result


def native(runtime,panel,update,full=False):
    rows=([r for r in panel['windows'] if r['role']=='monitor'] if full
          else select_windows(panel,runtime.config))
    reader=NativeInferenceReader(runtime.config,reader_manifest(rows))
    gates=runtime.gates(update,'binary');records=[]
    # CPU action targets are cached only in this runtime, which pins the teacher,
    # input manifest and inference settings. Never reuse across artifact identities.
    if not hasattr(runtime,'native_targets'):runtime.native_targets={}
    if not hasattr(runtime,'native_inputs'):runtime.native_inputs={}
    ref=runtime.reference
    def prefill(engine,gates):
        def run(inputs,noise,time,mask,*_):return engine(inputs,noise,time,mask,*gates,return_video=False)
        return run
    with diagnostic_mode(runtime):
        for row,valid in padded_rows(rows,runtime.rank,runtime.world):
            if row['id'] not in runtime.native_inputs:
                runtime.native_inputs[row['id']]=reader.read(row['episode'],row['frame'])
            inputs,target=runtime.native_inputs[row['id']]
            missing=torch.tensor(int(row['id'] not in runtime.native_targets),device=runtime.device)
            dist.all_reduce(missing,op=dist.ReduceOp.MAX)
            if missing:
                with masked_prefill(ref,prefill(runtime.teacher,(None,None,None))):
                    teacher=ref.infer_action(**inputs)['action'].float().cpu()
                runtime.native_targets[row['id']]=teacher
            teacher=runtime.native_targets[row['id']]
            with masked_prefill(ref,prefill(runtime.engine,gates)):
                student=ref.infer_action(**inputs)['action'].float().cpu()
            truth=torch.as_tensor(target['normalized']);n=min(24,target['valid_steps'])
            if student.shape!=(32,16) or n<1 or not torch.isfinite(student).all():
                raise FloatingPointError('Invalid native action output')
            if valid:
                records.append(dict(window_id=row['id'],episode=row['episode'],frame=row['frame'],
                    task=row['task'],suite=row['suite'],executed_steps=n,
                    teacher_error=float((teacher[:n]-truth[:n]).square().mean()),
                    student_error=float((student[:n]-truth[:n]).square().mean()),
                    student_teacher_error=float((student[:n]-teacher[:n]).square().mean()),
                    teacher_components=component_errors(teacher[:n][None],truth[:n][None],None)[0].tolist(),
                    student_components=component_errors(student[:n][None],truth[:n][None],None)[0].tolist()))
    records=gather_rows(records,rows)
    from collections import defaultdict
    task_agreement=defaultdict(list);suite_tasks=defaultdict(list)
    for r in records:task_agreement[r['task']].append(r['student_teacher_error'])
    agreement={'task/'+k:sum(v)/len(v) for k,v in task_agreement.items()}
    for task in task_agreement:
        suite=next(r['suite'] for r in records if r['task']==task)
        suite_tasks[suite].append(agreement['task/'+task])
    agreement.update({'suite/'+k:sum(v)/len(v) for k,v in suite_tasks.items()})
    agreement['overall']=sum(agreement['task/'+t] for t in task_agreement)/len(task_agreement)
    return dict(update=update,full=full,simulation=False,native_10_step_sampler=True,
                rows=records,actions=aggregate(records),action_agreement=agreement)


def flow(runtime,panel,update):
    rows=[r for r in panel['windows'] if r['role']=='monitor'];result={}
    with diagnostic_mode(runtime):
        for mode in (('soft','binary') if runtime.masks is not None else ('compact',)):
            records=[]
            for row,valid in padded_rows(rows,runtime.rank,runtime.world):
                loss,metrics,details,_=runtime.forward([row],[-2_000_000-row['id']],update,
                    mode=mode if mode!='compact' else 'soft')
                if valid:
                    records.append(dict(window_id=row['id'],task=row['task'],suite=row['suite'],
                        episode=row['episode'],frame=row['frame'],
                        student_error=float(details['student_error'][0]),teacher_error=float(details['teacher_error'][0]),
                        student_components=details['student_components'][0].tolist(),
                        teacher_components=details['teacher_components'][0].tolist(),
                        metrics={k:float(v) for k,v in metrics.items()}))
            gathered=gather_rows(records,rows)
            result[mode]=dict(rows=gathered,actions=aggregate(gathered))
    return result
