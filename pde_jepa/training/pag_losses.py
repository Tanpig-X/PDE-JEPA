# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""PAG geometry and normalized prediction losses."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def _validate_lags(
    lags: Sequence[int],
    lag_weights: Sequence[float],
    num_velocities: int,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    parsed_lags = tuple(int(lag) for lag in lags)
    parsed_weights = tuple(float(weight) for weight in lag_weights)
    if not parsed_lags or len(parsed_lags) != len(parsed_weights):
        raise ValueError("lags and lag_weights must be non-empty and have equal length")
    if len(set(parsed_lags)) != len(parsed_lags):
        raise ValueError("geometry lags must be unique")
    if any(lag < 1 or lag >= num_velocities for lag in parsed_lags):
        raise ValueError(
            f"Each lag must lie in [1, {num_velocities - 1}], received {parsed_lags}"
        )
    if any(weight < 0 for weight in parsed_weights) or not any(
        weight > 0 for weight in parsed_weights
    ):
        raise ValueError("lag_weights must be non-negative with at least one positive value")
    return parsed_lags, parsed_weights


def _velocity_direction_cosines(
    sequence: torch.Tensor,
    lags: Sequence[int],
    *,
    eps: float,
) -> dict[int, torch.Tensor]:
    if sequence.ndim != 3:
        raise ValueError(f"Expected sequence [B,T,F], received {tuple(sequence.shape)}")
    if sequence.shape[1] < 3:
        raise ValueError("Geometry matching requires at least three frames")
    velocity = sequence[:, 1:] - sequence[:, :-1]
    direction = F.normalize(velocity.float(), dim=-1, eps=eps)
    return {
        lag: (direction[:, :-lag] * direction[:, lag:]).sum(dim=-1)
        for lag in lags
    }


def temporal_geometry_matching_loss(
    pixels: torch.Tensor,
    latent: torch.Tensor,
    *,
    lags: Sequence[int] = (1, 2, 4),
    lag_weights: Sequence[float] = (1.0, 0.5, 0.25),
    loss: str = "smooth_l1",
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match multi-lag velocity-angle geometry to the real pixel trajectory.

    Pixel targets are always detached.  The returned loss has a gradient only
    through ``latent``.
    """

    if pixels.ndim != 5:
        raise ValueError(f"Expected pixels [B,C,T,H,W], received {tuple(pixels.shape)}")
    if latent.ndim != 4:
        raise ValueError(f"Expected latent [B,T,N,D], received {tuple(latent.shape)}")
    if pixels.shape[0] != latent.shape[0] or pixels.shape[2] != latent.shape[1]:
        raise ValueError(
            "Pixel and latent batch/time axes differ: "
            f"pixels={tuple(pixels.shape)}, latent={tuple(latent.shape)}"
        )
    if eps <= 0:
        raise ValueError("geometry eps must be positive")
    parsed_lags, parsed_weights = _validate_lags(
        lags,
        lag_weights,
        num_velocities=latent.shape[1] - 1,
    )
    if loss != "smooth_l1":
        raise ValueError(f"Only smooth_l1 geometry loss is supported, received {loss!r}")

    pixel_sequence = pixels.detach().permute(0, 2, 1, 3, 4).float().flatten(2)
    latent_sequence = latent.float().flatten(2)
    with torch.no_grad():
        pixel_cosines = _velocity_direction_cosines(
            pixel_sequence,
            parsed_lags,
            eps=eps,
        )
    latent_cosines = _velocity_direction_cosines(
        latent_sequence,
        parsed_lags,
        eps=eps,
    )

    total = latent_sequence.sum() * 0.0
    stats: dict[str, torch.Tensor] = {}
    for lag, weight in zip(parsed_lags, parsed_weights):
        target = pixel_cosines[lag].detach()
        predicted = latent_cosines[lag]
        lag_loss = F.smooth_l1_loss(predicted, target)
        total = total + weight * lag_loss
        stats[f"pixel_cos_lag{lag}"] = target.mean().detach()
        stats[f"latent_cos_lag{lag}"] = predicted.mean().detach()
        stats[f"cos_gap_lag{lag}"] = (predicted - target).abs().mean().detach()
        stats[f"loss_lag{lag}"] = lag_loss.detach()
    stats["geometry_loss"] = total.detach()
    return total, stats


def normalized_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {prediction.shape} vs {target.shape}"
        )
    numerator = (prediction.float() - target.float()).pow(2).mean()
    denominator = target.float().pow(2).mean().detach().clamp_min(eps)
    return numerator / denominator

