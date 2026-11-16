"""Shared immutable weights, separate teacher/student states and differentiable actions."""
import time
import torch
from vpp2.compact_kv.model import load_reference,action_prediction
from vpp2.full_mask.data import WindowReader
from vpp2.full_mask.runtime import distributed,barrier,gather,model_versions,check_frozen
from .controller import Controller
from .geometry import Geometry,ParameterInventory
from .masking import install
from .data import future_mask
from .losses import objectives,cache_terms

def initialize(config,install_branch=True):
    model=load_reference(config)
    if any(isinstance(m,torch.nn.Dropout) and m.p>0 for m in model.modules()):raise ValueError('Unexpected active dropout')
    from .execution import configure_recomputation
    model.mot.training=True;configure_recomputation(model,config)
    reader=WindowReader(config);model._full_mask_conditioning_cache=reader.runtime_cache
    geometry=Geometry(**config['geometry']);inventory=ParameterInventory.from_model(model.video_expert,geometry)
    if inventory.original!=config['original_video_parameters']:raise ValueError('Original parameter count changed')
    branch=install(model.video_expert) if install_branch else None
    if branch is not None:branch.train()
    controller=Controller(geometry,config['seed'],learn_hidden=not config['budget']['protect_hidden']).cuda()
    cache_masks=config.get('execution',{}).get('cache_hard_masks',False)
    if not isinstance(cache_masks,bool):raise ValueError('Hard-mask caching must be boolean')
    controller.cache_hard_masks=cache_masks
    return model,reader,controller,branch,inventory

def forward_video(branch,parts,gates=(None,None,None)):
    return branch(parts[0],parts[1],parts[2],parts[6][:1950,:1950],*gates)

def predict(model,parts,cache):return action_prediction(model,parts[0],cache,parts[3],parts[4],parts[6])

def step(model,branch,controller,parts,rows,pruning_updates,config):
    begin=time.perf_counter()
    with torch.no_grad():
        teacher=forward_video(branch,parts);teacher_action=predict(model,parts,teacher['cache'])
    teacher_seconds=time.perf_counter()-begin;begin=time.perf_counter()
    student=forward_video(branch,parts,controller.forward_gates());student_action=predict(model,parts,student['cache'])
    student_seconds=time.perf_counter()-begin
    terms=cache_terms(student['cache'],teacher['cache'])
    loss,metrics,details,objectives_=objectives(student_action,teacher_action,parts[5],parts[0]['action_loss_is_pad'],
        model.train_action_scheduler.training_weight(parts[4]),student['video'],teacher['video'],terms,
        future_mask(rows,parts[1].device),pruning_updates,
        kv_weight=config['train']['kv_weight'],video_weight=config['train']['teacher_video_weight'],ramp_updates=config['train']['teacher_ramp_updates'])
    metrics.update({'timing/teacher_host_seconds':teacher_seconds,'timing/student_host_seconds':student_seconds})
    return loss,metrics,details,objectives_
