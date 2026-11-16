"""One detached per-window reliability weight for teacher action and KV matching."""
import numpy as np
import torch
from vpp2.compact_kv.losses import action_error


def thresholds_from_rows(rows, *, expected_counts=None, minimum_count=64, expected_tasks=None):
    """Keep the legacy 64-draw contract unless an explicit sampling plan is supplied."""
    tasks=sorted(expected_tasks if expected_tasks is not None else {r['task'] for r in rows})
    if not tasks or {r['task'] for r in rows} != set(tasks):
        raise ValueError('Calibration tasks do not match the sampling plan')
    if any(r['stratum'] not in range(4) for r in rows):
        raise ValueError('Invalid action-noise stratum')
    required={(task,s) for task in tasks for s in range(4)}
    if expected_counts is not None and set(expected_counts)!=required:
        raise ValueError('Calibration count plan must cover every task/noise stratum')
    result={}
    for task in tasks:
        result[task]=[]
        for stratum in range(4):
            values=[r['teacher_error'] for r in rows if r['task']==task and r['stratum']==stratum]
            expected=64 if expected_counts is None else expected_counts[(task,stratum)]
            if expected < minimum_count or len(values)!=expected:
                raise ValueError(f'Calibration count mismatch for {task}/{stratum}: {len(values)} != {expected}; minimum {minimum_count}')
            if not np.isfinite(values).all() or any(x < 0 for x in values):
                raise ValueError('Calibration errors must be finite and nonnegative')
            median,q75,q95=np.quantile(values,[.5,.75,.95])
            result[task].append(dict(median=float(median),q75=float(q75),q95=float(q95),count=len(values)))
    return result


def weights(teacher_error, student_error, zero_error, task_ids, strata, thresholds):
    if any(not torch.isfinite(x).all() for x in (teacher_error,student_error,zero_error)):
        raise FloatingPointError('Nonfinite action predictions/targets cannot be treated as teacher unreliability')
    with torch.no_grad():
        table = torch.tensor([[r['median'],r['q75'],r['q95']] for t,s in zip(task_ids,strata)
                              for r in [thresholds[t][int(s)]]], device=teacher_error.device, dtype=torch.float32)
        median, low, high = table.unbind(-1)
        # The <= branch deliberately returns one for perfect/degenerate teachers.
        reliability = torch.where(teacher_error <= low, 1.,
            ((high-teacher_error)/(high-low).clamp_min(1e-8)).clamp(0.,1.))
        delta = (.01*median).clamp_min(1e-8)
        better_student = teacher_error > student_error + delta
        worse_than_zero = teacher_error > zero_error + delta
        applied = reliability * (~better_student & ~worse_than_zero)
    return applied, dict(reliability=reliability, better_student=better_student, worse_than_zero=worse_than_zero)


def action_losses(student, teacher, target, mask, time_weight, task_ids, strata, thresholds):
    e_student = action_error(student,target,mask)
    e_teacher = action_error(teacher,target,mask)
    e_zero = action_error(torch.zeros_like(target),target,mask)
    w, reasons = weights(e_teacher.detach(),e_student.detach(),e_zero,task_ids,strata,thresholds)
    raw = action_error(student,teacher,mask)*time_weight
    # Mean over the full batch, never divide by the sum of weights.
    return (e_student*time_weight).mean(), (raw*w).mean(), raw.mean(), w, reasons


def kv_losses(terms, teacher_weights):
    """Raw diagnostic, gated objective and gated per-layer diagnostics.

    terms has one teacher-energy-normalized K/V error per window and native
    layer. Use the SAME detached weights returned by action_losses. Rejected
    windows contribute zero matching gradients at every interface; ground-truth
    action loss is deliberately outside this function and remains active.
    Average over the full batch, never over the accepted subset or weight sum.
    """
    if terms.ndim != 3 or tuple(terms.shape[1:]) != (40, 2) or not terms.shape[0]:
        raise ValueError('Expected nonempty per-window KV terms [B,40,2]')
    if teacher_weights.shape != terms.shape[:1]:
        raise ValueError('KV reliability must have exactly one weight per window')
    values = terms.float()
    weight = teacher_weights.detach().to(device=terms.device, dtype=torch.float32)
    if not torch.isfinite(values).all() or not torch.isfinite(weight).all():
        raise FloatingPointError('Nonfinite KV terms or reliability weights')
    if not ((weight >= 0) & (weight <= 1)).all():
        raise ValueError('Teacher reliability weights must be in [0,1]')
    applied = values * weight[:, None, None]
    return values.mean(), applied.mean(), applied.detach().mean(0)
