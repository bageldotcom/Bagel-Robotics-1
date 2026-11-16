"""Named per-window losses; reliability never changes the batch denominator."""
import torch
from vpp2.compact_kv.losses import action_error
from .reliability import weights

def coordinate_max_error(prediction,target,is_pad):
    err=(prediction.float()-target.detach().float()).square()
    valid=torch.ones_like(err[...,0]) if is_pad is None else (~is_pad.bool()).float()
    if (valid.sum(-1)==0).any():raise ValueError('No valid actions')
    return ((err*valid[...,None]).sum(1)/valid.sum(1,keepdim=True)).max(-1).values

def component_errors(prediction,target,is_pad):
    # EE16: XYZ position, quaternion coordinates and gripper for both arms.
    # Tiny algebra tests may use fewer coordinates; then use one complete group.
    groups=([0,1,2,8,9,10],[3,4,5,6,11,12,13,14],[7,15]) if prediction.shape[-1]==16 else (list(range(prediction.shape[-1])),)
    return torch.stack([action_error(prediction[...,g],target[...,g],is_pad) for g in groups],-1)

def video_error(student,teacher,valid):
    if student.shape!=teacher.shape or student.ndim!=5:raise ValueError('Video shape mismatch')
    # Native temporal positions 3 and 4 only. Mask whole incomplete future planes.
    a=student[:,:,3:].float();b=teacher.detach()[:,:,3:].float()
    if valid.shape!=(len(a),2):raise ValueError('Expected two future-plane masks')
    error=(a-b).square().mean((1,3,4));n=valid.sum(-1)
    return (error*valid).sum(-1)/n.clamp_min(1),n>0

def objectives(student_action,teacher_action,target,is_pad,time_weight,student_video,teacher_video,
               cache_errors,valid_future,pruning_updates,kv_weight=.05,video_weight=.05,ramp_updates=100):
    if cache_errors.shape!=(len(target),40,2):raise ValueError('All 40 cache slots required')
    gt=action_error(student_action,target,is_pad);te=action_error(teacher_action,target,is_pad)
    ze=action_error(torch.zeros_like(target),target,is_pad)
    ce=coordinate_max_error(teacher_action,target,is_pad)
    teacher_components=component_errors(teacher_action,target,is_pad)
    student_components=component_errors(student_action,target,is_pad)
    reliability,why=weights(te.detach(),gt.detach(),ze.detach(),
        teacher_components,component_errors(torch.zeros_like(target),target,is_pad))
    vp,vvalid=video_error(student_video,teacher_video,valid_future)
    late=cache_errors[:,20:40].mean((1,2))
    ramp=min(max(pruning_updates,0)/ramp_updates,1.)
    terms=dict(gt_action=(gt*time_weight.float().reshape(-1)).mean(),
               late_kv=(reliability*late).mean(),teacher_video=(reliability*vp).mean())
    total=terms['gt_action']+ramp*kv_weight*terms['late_kv']+ramp*video_weight*terms['teacher_video']
    metrics={'schema_version':3,'loss/total':total.detach(),'loss/gt_action':terms['gt_action'].detach(),
        'loss/late_kv_raw':late.detach().mean(),'loss/late_kv_gated':terms['late_kv'].detach(),
        'loss/late_kv_applied':(ramp*kv_weight*terms['late_kv']).detach(),
        'loss/video_raw':vp.detach().mean(),'loss/video_gated':terms['teacher_video'].detach(),
        'loss/video_applied':(ramp*video_weight*terms['teacher_video']).detach(),
        'loss/teacher_action_weight':0.,'loss/budget_weight':0.,'loss/entropy_weight':0.,'loss/discreteness_weight':0.,
        'reliability/base':why['base'].mean(),'reliability/applied':reliability.mean(),
        'reliability/student_veto':why['student_veto'].float().mean(),'reliability/zero_veto':why['zero_veto'].float().mean(),
        'reliability/component_veto':why['component_veto'].float().mean(),'supervision/ramp':ramp,
        'supervision/effective_kv':ramp*kv_weight*reliability.mean(),
        'supervision/effective_video':ramp*video_weight*(reliability*vvalid).mean()}
    for layer in range(40):
        for k,key in enumerate(('k','v')):
            metrics[f'kv/layer_{layer:02d}_{key}']=cache_errors[:,layer,k].detach().mean()
    details=dict(student_components=student_components,teacher_components=teacher_components,student_error=gt,teacher_error=te,zero_error=ze,coordinate_error=ce,video_error=vp,
                 base_reliability=why['base'],reliability=reliability,video_valid=vvalid,**{k:v for k,v in why.items() if k!='base'})
    return total,metrics,details,terms


def cache_terms(student,teacher):
    """Early-layer values are diagnostics; only late-layer terms retain a graph."""
    from vpp2.compact_kv.losses import kv_terms
    if len(student)!=40 or len(teacher)!=40:raise ValueError('Expected all 40 cache slots')
    with torch.no_grad():early=kv_terms(student[:20],teacher[:20])
    late=kv_terms(student[20:],teacher[20:])
    return torch.cat([early,late],dim=1)
