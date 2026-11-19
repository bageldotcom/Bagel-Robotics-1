"""Per-window action/video errors and layer-wise cache comparisons."""
import torch
from bagel_robotics._internal.compact_kv.losses import action_error

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

def cache_terms(student,teacher):
    """Early-layer values are diagnostics; only late-layer terms retain a graph."""
    from bagel_robotics._internal.compact_kv.losses import kv_terms
    if len(student)!=40 or len(teacher)!=40:raise ValueError('Expected all 40 cache slots')
    with torch.no_grad():early=kv_terms(student[:20],teacher[:20])
    late=kv_terms(student[20:],teacher[20:])
    return torch.cat([early,late],dim=1)
