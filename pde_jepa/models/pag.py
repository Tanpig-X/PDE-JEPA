# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license in the repository LICENSE file.

import torch
import torch.nn as nn


class PhysicsAlignedGeometry(nn.Module):
    """Identity-initialized token-wise residual MLP.

    The module deliberately has no spatial or temporal token mixing.  It can
    only learn a shared channel-coordinate correction on top of the frozen
    encoder representation.
    """

    def __init__(
        self,
        dim: int = 192,
        hidden_dim: int = 384,
        *,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        if dim < 1 or hidden_dim < 1:
            raise ValueError("dim and hidden_dim must be positive")
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.norm = nn.LayerNorm(self.dim)
        self.fc1 = nn.Linear(self.dim, self.hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_dim, self.dim)
        if zero_init_output:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4:
            raise ValueError(
                "PhysicsAlignedGeometry expects [B,T,N,D], "
                f"received {tuple(latent.shape)}"
            )
        if latent.shape[-1] != self.dim:
            raise ValueError(
                f"Expected latent dimension {self.dim}, received {latent.shape[-1]}"
            )
        correction = self.fc2(self.act(self.fc1(self.norm(latent))))
        return latent + correction

