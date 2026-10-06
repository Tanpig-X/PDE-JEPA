"""Condition normalization and teacher forcing for the PAG auxiliary model."""

import torch
from torch import nn


def normalize_condition(parameters: torch.Tensor, spec: dict) -> torch.Tensor:
    """Normalize PDE conditions using statistics fitted on the training split."""
    value = parameters.float()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("PDE conditions must be finite")
    kind = spec.get("type", "log_standard" if "log_mean" in spec else "identity")
    if kind == "identity":
        return value
    if kind not in ("standard", "log_standard"):
        raise ValueError(f"Unknown condition normalization {kind!r}")
    center = torch.as_tensor(spec.get("mean", spec.get("log_mean", 0.0)), device=value.device, dtype=value.dtype)
    scale = torch.as_tensor(spec.get("std", spec.get("log_std", 1.0)), device=value.device, dtype=value.dtype)
    if (not bool(torch.isfinite(center).all()) or not bool(torch.isfinite(scale).all())
            or bool(torch.any(scale <= 0))):
        raise ValueError("Condition means must be finite and standard deviations positive")
    for statistic in (center, scale):
        if statistic.ndim > 1 or statistic.numel() not in (1, value.shape[-1]):
            raise ValueError("Normalization statistics must be scalar or have one value per condition")
    if kind == "log_standard":
        if bool(torch.any(value <= 0)):
            raise ValueError("Log-standardized conditions must be positive")
        value = torch.log(value)
    return (value - center) / scale



def predictor_teacher_deltas(
    predictor: nn.Module,
    projected_latent: torch.Tensor,
    condition: torch.Tensor,
) -> torch.Tensor:
    if projected_latent.ndim != 4:
        raise ValueError("projected_latent must have shape [B,T,N,D]")
    batch_size, num_frames, tokens, dim = projected_latent.shape
    transitions = num_frames - 1
    if condition.ndim == 2 and condition.shape[0] == batch_size:
        conditions = condition[:, None, :].expand(batch_size, transitions, -1)
    elif condition.ndim == 3 and condition.shape[:2] == (batch_size, transitions):
        conditions = condition
    else:
        raise ValueError("Conditions must have shape [B,P] or [B,T-1,P]")
    source = projected_latent[:, :-1].reshape(
        batch_size,
        transitions * tokens,
        dim,
    )
    prediction = predictor(source, conditions)
    expected = (batch_size, transitions * tokens, dim)
    if tuple(prediction.shape) != expected:
        raise RuntimeError(
            f"Delta predictor returned {tuple(prediction.shape)}, expected {expected}"
        )
    return prediction.reshape(batch_size, transitions, tokens, dim)


