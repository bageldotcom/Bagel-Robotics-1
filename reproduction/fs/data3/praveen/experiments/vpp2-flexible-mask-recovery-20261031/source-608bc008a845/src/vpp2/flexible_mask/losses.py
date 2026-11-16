"""GT action + reliable teacher future-video and late-K/V supervision."""
import torch
from vpp2.compact_kv.losses import action_error
from vpp2.adaptive_mask.losses import component_errors, video_error, cache_terms


@torch.no_grad()
def reliability(teacher_error, student_error, zero_error, teacher_components, zero_components):
    values = (teacher_error, student_error, zero_error, teacher_components, zero_components)
    if any(not torch.isfinite(v).all() or (v < 0).any() for v in values):
        raise FloatingPointError('Reliability needs finite nonnegative errors')
    base = (1 - teacher_error / student_error.clamp_min(1e-8)).clamp(0, 1)
    base = torch.where((teacher_error <= 1e-8) & (student_error <= 1e-8), 0., base)
    zero_veto = teacher_error > zero_error + 1e-8
    margins = (.01*zero_components).clamp_min(1e-8)
    component_veto = (teacher_components > zero_components + margins).any(-1)
    weight = base.masked_fill(zero_veto | component_veto, 0.)
    return weight, dict(base=base, zero_veto=zero_veto, component_veto=component_veto)


def objectives(student_action, teacher_action, target, is_pad, time_weight,
               student_video, teacher_video, kv, valid_future, teacher_ramp):
    if kv.shape != (len(target), 40, 2):
        raise ValueError('Expected all forty action-facing K/V slots')
    gt = action_error(student_action, target, is_pad)
    te = action_error(teacher_action.detach(), target, is_pad)
    zero = torch.zeros_like(target)
    sc = component_errors(student_action, target, is_pad)
    tc = component_errors(teacher_action.detach(), target, is_pad)
    r, reason = reliability(te, gt.detach(), action_error(zero, target, is_pad), tc, component_errors(zero, target, is_pad))
    video, valid = video_error(student_video, teacher_video, valid_future)
    late = kv[:, 20:40].mean((1,2))
    terms = dict(gt=(gt*time_weight.float().reshape(-1)).mean(),
                 video=(r*video).mean(), kv=(r*late).mean())
    loss = terms['gt'] + .05*teacher_ramp*(terms['video']+terms['kv'])
    metrics = dict(gt_action=terms['gt'].detach(), teacher_video_raw=video.detach().mean(),
                   teacher_video_gated=terms['video'].detach(), late_kv_raw=late.detach().mean(),
                   late_kv_gated=terms['kv'].detach(), reliability=r.mean(), rejected=(r==0).float().mean(),
                   zero_veto=reason['zero_veto'].float().mean(), component_veto=reason['component_veto'].float().mean(),
                   teacher_video_weight=.05*teacher_ramp, late_kv_weight=.05*teacher_ramp,
                   effective_video_weight=.05*teacher_ramp*(r*valid).mean(), effective_kv_weight=.05*teacher_ramp*r.mean())
    for layer in range(40):
        for j, name in enumerate(('k','v')):
            metrics[f'kv/layer_{layer:02d}_{name}'] = kv[:,layer,j].detach().mean()
    return loss, metrics, dict(student_error=gt.detach(),teacher_error=te.detach(),
        student_components=sc.detach(),teacher_components=tc.detach(),reliability=r), terms
