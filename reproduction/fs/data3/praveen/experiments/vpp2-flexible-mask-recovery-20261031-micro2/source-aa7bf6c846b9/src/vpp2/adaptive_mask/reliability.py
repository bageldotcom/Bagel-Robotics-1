"""Calibration-free, detached demonstration-error weights, computed in the training batch."""
import torch

@torch.no_grad()
def weights(teacher,student,zero,teacher_groups=None,zero_groups=None,epsilon=1e-8,margin_fraction=.01):
    if teacher.ndim!=1 or student.shape!=teacher.shape or zero.shape!=teacher.shape:raise ValueError('Reliability shape mismatch')
    vectors=[teacher,student,zero]+([] if teacher_groups is None else [teacher_groups,zero_groups])
    if any(x is None or not torch.isfinite(x).all() or (x<0).any() for x in vectors):raise FloatingPointError('Invalid action errors')
    scale=zero.clamp_min(epsilon)
    # The zero-flow baseline is computed from the SAME target, with no model
    # forward or fitting pass. This is relative usefulness, not calibrated probability.
    base=(1-teacher/scale).clamp(0,1)
    margin=(margin_fraction*scale).clamp_min(epsilon)
    student_veto=teacher>student+margin
    zero_veto=teacher>zero+epsilon
    group_veto=torch.zeros_like(student_veto)
    if teacher_groups is not None:
        if teacher_groups.shape!=zero_groups.shape or teacher_groups.shape[0]!=len(teacher):raise ValueError('Component error shape mismatch')
        group_margin=(margin_fraction*zero_groups).clamp_min(epsilon)
        group_veto=(teacher_groups>zero_groups+group_margin).any(-1)
    applied=base.masked_fill(student_veto|zero_veto|group_veto,0)
    return applied,dict(base=base,student_veto=student_veto,zero_veto=zero_veto,component_veto=group_veto)
