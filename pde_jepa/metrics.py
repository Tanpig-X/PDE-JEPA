"""Relative L2 training loss and physical-space NRMSE metrics."""

import torch


def relative_l2(prediction, target, eps=1e-8, reduction="mean"):
    """Average relative L2 over channels and samples; all times are joint."""
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")
    error = (prediction.float() - target.float()).flatten(2)
    truth = target.float().flatten(2)
    value = (torch.linalg.vector_norm(error, dim=-1) /
             (torch.linalg.vector_norm(truth, dim=-1) + eps)).mean(1)
    return value.mean() if reduction == "mean" else value


def nrmse(prediction, target, eps=1e-8):
    """Return [batch,channel] errors; callers choose the time aggregation."""
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")
    error = (prediction.float() - target.float()).flatten(2)
    truth = target.float().flatten(2)
    count = torch.full_like(error[..., 0], error.shape[-1])
    return (error.square().sum(-1) / count).sqrt() / ((truth.square().sum(-1) / count).sqrt() + eps)


def trajectory_metrics(prediction, target, channel_std=1.0, eps=1e-8,
                       channel_mean=0.0, observed_frames=1):
    """Compute physical-space errors over all frames and forecast frames."""
    if not 1 <= observed_frames < target.shape[2]:
        raise ValueError("observed_frames must leave at least one forecast frame")
    shape = (1, -1) + (1,) * (target.ndim - 2)
    scale = torch.as_tensor(channel_std, device=target.device, dtype=torch.float32).reshape(shape)
    center = torch.as_tensor(channel_mean, device=target.device, dtype=torch.float32).reshape(shape)
    prediction, target = prediction.float() * scale + center, target.float() * scale + center
    per_step = torch.stack([nrmse(prediction[:, :, t], target[:, :, t], eps).mean(1)
                            for t in range(target.shape[2])], 1)
    return {
        "all_frame_nrmse": nrmse(prediction, target, eps).mean(1),
        "future_mean_step_nrmse": per_step[:, observed_frames:].mean(1),
        "future_joint_nrmse": nrmse(prediction[:, :, observed_frames:], target[:, :, observed_frames:], eps).mean(1),
        "last_step_nrmse": per_step[:, -1],
    }
