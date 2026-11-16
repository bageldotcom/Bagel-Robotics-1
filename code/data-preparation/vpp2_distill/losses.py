"""All valid demonstration actions; optional teacher responses and structured K/V."""
import torch


def action_mse(prediction, target, is_pad):
    if prediction.shape != target.shape or is_pad.shape != prediction.shape[:2]:
        raise ValueError("Action loss shapes differ")
    valid = (~is_pad.bool()).float()
    # Mask before subtraction: padded labels may contain non-finite sentinels.
    delta = torch.where(valid[..., None].bool(), prediction.float() - target.float(), 0.0)
    return (delta.square().mean(-1) * valid).sum(-1) / valid.sum(-1).clamp_min(1)


def structured_kv_loss(student, target, batch_index):
    errors = []
    for i, layer in enumerate(target["layers"].tolist()):
        positions = target["tokens"][i].to(student[layer]["k"].device)
        for kind in ("k", "v"):
            actual = student[layer][kind][batch_index].index_select(0, positions).float()
            expected = target[kind][i].to(actual.device).detach().float()
            if actual.shape != expected.shape:
                raise ValueError("Teacher K/V target shape mismatch")
            errors.append((actual - expected).square().mean() / (expected.square().mean() + 1e-6))
    if not errors:
        raise ValueError("An alignment target cannot contain zero layers")
    return torch.stack(errors).mean()


def policy_losses(prediction, gt_target, is_pad, times, scheduler, *, gt_valid=None,
                  teacher_velocity=None, teacher_valid=None, memories=None, kv_targets=None,
                  lambda_kd=0.5, lambda_kv=0.1, gt_scale=1.0, teacher_scale=1.0):
    batch = prediction.shape[0]
    gt_valid = torch.ones(batch, device=prediction.device, dtype=torch.bool) if gt_valid is None else gt_valid.bool()
    teacher_valid = torch.zeros_like(gt_valid) if teacher_valid is None else teacher_valid.bool()
    per_gt = action_mse(prediction, gt_target, is_pad) * scheduler.training_weight(times).to(prediction.device)
    # Scales are supplied by the logical-batch sampler, not local microbatch counts.
    # This keeps distributed accumulation equivalent to the specified global means.
    gt = (per_gt * gt_valid).mean() * gt_scale
    zero = prediction.float().sum() * 0
    kd = zero
    kv = zero
    if teacher_valid.any():
        if teacher_velocity is None:
            raise ValueError("Missing teacher action-output labels")
        kd = (action_mse(prediction, teacher_velocity.detach(), is_pad) * teacher_valid).mean() * teacher_scale
    if kv_targets is not None:
        terms = [structured_kv_loss(memories, target, i) for i, target in enumerate(kv_targets) if target is not None]
        if terms:
            kv = torch.stack(terms).sum() / batch * teacher_scale
    total = gt + lambda_kd * kd + lambda_kv * kv
    return {"action_loss": total, "gt_loss": gt.detach(), "teacher_action_loss": kd.detach(), "kv_loss": kv.detach()}
