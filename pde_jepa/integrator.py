"""Small differentiable fixed-step classical RK4 solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class RK4Diagnostics:
    max_substep_increment_rms: torch.Tensor
    max_intermediate_state_rms: torch.Tensor
    nonfinite_count: torch.Tensor

    def detached_dict(self) -> dict[str, float | int]:
        return {
            "max_rk_substep_increment_rms": float(
                self.max_substep_increment_rms.detach().float().cpu()
            ),
            "max_rk_intermediate_state_rms": float(
                self.max_intermediate_state_rms.detach().float().cpu()
            ),
            "nan_inf_count": int(self.nonfinite_count.detach().cpu()),
        }


def _rms(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(value.float().square()))


def fixed_step_rk4(
    field: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    q0: torch.Tensor,
    condition: torch.Tensor,
    *,
    delta_t: float | torch.Tensor,
    substeps: int,
    return_diagnostics: bool = False,
    derivative_dtype: str = "float32",
    step_size_mode: str = "scalar",
) -> torch.Tensor | tuple[torch.Tensor, RK4Diagnostics]:
    """Integrate one data interval without detaching any intermediate state.

    The field callable may internally use mixed precision. Every returned
    derivative is promoted to FP32 by default. ``native`` retains the
    vector field's dtype for weighted derivatives while keeping states FP32.
    """

    if q0.ndim < 2:
        raise ValueError("q0 must include batch and state dimensions")
    if condition.shape[0] != q0.shape[0]:
        raise ValueError("condition batch dimension does not match q0")
    if not isinstance(substeps, int) or substeps < 1:
        raise ValueError("substeps must be a positive integer")
    intervals = torch.as_tensor(delta_t, device=q0.device, dtype=torch.float32)
    if not bool(torch.isfinite(intervals).all()) or bool((intervals <= 0).any()):
        raise ValueError("delta_t must be finite and positive")
    if intervals.numel() not in (1, q0.shape[0]):
        raise ValueError('delta_t must be scalar or contain one interval per batch element')

    if derivative_dtype not in ("float32", "native"):
        raise ValueError("derivative_dtype must be float32 or native")
    q = q0.float()
    if step_size_mode not in ('scalar', 'batch_tensor'):
        raise ValueError('step_size_mode must be scalar or batch_tensor')
    if step_size_mode == 'batch_tensor' or intervals.numel() > 1:
        h = intervals.reshape(-1).expand(q.shape[0]).reshape((q.shape[0],) + (1,) * (q.ndim - 1)) / substeps
    else:
        h = (intervals.reshape(()) / substeps if derivative_dtype == 'native'
             else float(delta_t) / substeps)
    def derivative(state):
        value = field(state, condition)
        return value.float() if derivative_dtype == "float32" else value
    max_increment = torch.zeros((), device=q.device, dtype=torch.float32)
    max_state = _rms(q)
    nonfinite = torch.count_nonzero(~torch.isfinite(q))

    for _ in range(substeps):
        k1 = derivative(q)
        q2 = q + (0.5 * h) * k1
        k2 = derivative(q2)
        q3 = q + (0.5 * h) * k2
        k3 = derivative(q3)
        q4 = q + h * k3
        k4 = derivative(q4)
        increment = (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        q = q + increment

        max_increment = torch.maximum(max_increment, _rms(increment))
        for value in (q2, q3, q4, q):
            max_state = torch.maximum(max_state, _rms(value))
            nonfinite = nonfinite + torch.count_nonzero(~torch.isfinite(value))
        for value in (k1, k2, k3, k4):
            nonfinite = nonfinite + torch.count_nonzero(~torch.isfinite(value))

    if return_diagnostics:
        return q, RK4Diagnostics(max_increment, max_state, nonfinite)
    return q
