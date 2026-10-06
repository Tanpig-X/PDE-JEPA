"""Task-specific Physics-Structured Predictors (PSP), paper Section 4.3.

Construct a predictor with ``build_psp(task, **kwargs)``.
Vorticity, Advection, Wave-2D, Gray–Scott, Wave-B and Combined return latent
time derivatives. Burgers/Heat conditional Markov models return the next state.
See README for task-specific states and conditions.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from pde_jepa.models.auxiliary_predictor import AuxiliaryDynamicsPredictor
from pde_jepa.models.spatial_blocks import (
    ACBlock, Block, MLP, build_action_block_causal_attention_mask, rotate_queries_or_keys,
)
from pde_jepa.models.positions import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed
from pde_jepa.utils.tensors import trunc_normal_


@dataclass(frozen=True)
class RawNuStats:
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("Raw-viscosity bounds must be finite")
        if self.maximum <= self.minimum:
            raise ValueError("Raw-viscosity maximum must exceed minimum")

    @property
    def center(self) -> float:
        return 0.5 * (self.minimum + self.maximum)

    @property
    def scale(self) -> float:
        return 0.5 * (self.maximum - self.minimum)

    def transform(self, nu: torch.Tensor) -> torch.Tensor:
        value = nu.float()
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Raw viscosity contains NaN or Inf")
        return (value - self.center) / self.scale

    def as_dict(self) -> dict[str, float]:
        return {
            "nu_min_train": self.minimum,
            "nu_max_train": self.maximum,
            "nu_center": self.center,
            "nu_scale": self.scale,
        }


class VorticityPhysicsStructuredPredictor(nn.Module):
    """Spatial Transformer implementing ``C(q) + r_nu D(q)``."""

    def __init__(
        self,
        *,
        input_dim: int = 192,
        embed_dim: int = 384,
        depth: int = 24,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        token_grid_size: int = 16,
        use_sdpa: bool = True,
        activation_checkpointing: bool = False,
        init_std: float = 0.02,
        head_init_std: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if input_dim < 1 or embed_dim < 1 or depth < 1 or num_heads < 1:
            raise ValueError("Model dimensions/depth/head count must be positive")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if token_grid_size < 1:
            raise ValueError("token_grid_size must be positive")
        if dropout != 0.0 or attention_dropout != 0.0:
            raise ValueError("PSP uses zero dropout")
        if not math.isfinite(head_init_std) or head_init_std < 0:
            raise ValueError("head_init_std must be finite and non-negative")

        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.token_grid_size = int(token_grid_size)
        self.num_tokens = self.token_grid_size**2
        self.activation_checkpointing = bool(activation_checkpointing)
        self.init_std = float(init_std)
        self.head_init_std = float(head_init_std)
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.input_projection = nn.Linear(self.input_dim, self.embed_dim)
        position = get_2d_sincos_pos_embed(
            self.embed_dim,
            self.token_grid_size,
            cls_token=False,
        )
        if position.shape != (self.num_tokens, self.embed_dim):
            raise RuntimeError(f"Unexpected positional encoding shape {position.shape}")
        self.register_buffer(
            "spatial_positional_embedding",
            torch.from_numpy(np.asarray(position, dtype=np.float32)).unsqueeze(0),
            persistent=True,
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=float(mlp_ratio),
                    qkv_bias=bool(qkv_bias),
                    drop=float(dropout),
                    attn_drop=float(attention_dropout),
                    drop_path=0.0,
                    use_sdpa=bool(use_sdpa),
                    is_causal=False,
                    use_rope=False,
                )
                for _ in range(self.depth)
            ]
        )
        self.c_norm = nn.LayerNorm(self.embed_dim)
        self.c_projection = nn.Linear(self.embed_dim, self.input_dim)
        self.d_norm = nn.LayerNorm(self.embed_dim)
        self.d_projection = nn.Linear(self.embed_dim, self.input_dim)
        self.apply(self._init_weights)
        self._rescale_blocks()
        trunc_normal_(self.c_projection.weight, std=self.head_init_std)
        trunc_normal_(self.d_projection.weight, std=self.head_init_std)
        nn.init.zeros_(self.c_projection.bias)
        nn.init.zeros_(self.d_projection.bias)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _rescale_blocks(self) -> None:
        for layer_index, block in enumerate(self.blocks, start=1):
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * layer_index))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * layer_index))

    def forward_hidden(self, q: torch.Tensor) -> torch.Tensor:
        if q.ndim != 3:
            raise ValueError(f"q must be [B,N,D], received {tuple(q.shape)}")
        if q.shape[1:] != (self.num_tokens, self.input_dim):
            raise ValueError(
                f"Expected q trailing shape {(self.num_tokens, self.input_dim)}, "
                f"received {tuple(q.shape[1:])}"
            )
        hidden = self.input_projection(self.input_norm(q))
        hidden = hidden + self.spatial_positional_embedding.to(dtype=hidden.dtype)
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                hidden = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden,
                    use_reentrant=False,
                )
            else:
                hidden = block(hidden)
        return hidden

    def forward_fields(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.forward_hidden(q)
        return self.c_projection(self.c_norm(hidden)), self.d_projection(self.d_norm(hidden))

    def structured_generator(
        self, q: torch.Tensor, r_nu: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if r_nu.ndim == 1:
            r_nu = r_nu.unsqueeze(-1)
        if r_nu.shape != (q.shape[0], 1):
            raise ValueError(
                f"Expected centered raw-nu coordinate {(q.shape[0], 1)}, "
                f"received {tuple(r_nu.shape)}"
            )
        if not bool(torch.isfinite(r_nu).all()):
            raise ValueError("Centered raw-nu coordinate contains NaN or Inf")
        c_field, d_field = self.forward_fields(q)
        scaled_d = r_nu.to(dtype=d_field.dtype).unsqueeze(-1) * d_field
        return c_field + scaled_d, c_field, d_field, scaled_d

    def forward(self, q: torch.Tensor, r_nu: torch.Tensor, *, return_fields: bool = False):
        field, c_field, d_field, scaled_d = self.structured_generator(q, r_nu)
        if return_fields:
            return field, c_field, d_field, scaled_d
        return field


class _WaveRoPEAttention(nn.Module):
    """Three-axis RoPE on one latent frame, including rectangular grids."""

    def __init__(self, dim, num_heads, qkv_bias=True, use_sdpa=True, grid_size=16):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(dim) // self.num_heads
        self.axis_dim = 2 * ((self.head_dim // 3) // 2)
        if self.axis_dim < 4:
            raise ValueError('RoPE requires at least four features per axis')
        self.scale = self.head_dim ** -0.5
        self.grid_size = int(grid_size)
        self.use_sdpa = bool(use_sdpa)
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, *, T=1, H_patches=None, W_patches=None):
        batch, tokens, dim = x.shape
        height = self.grid_size if H_patches is None else int(H_patches)
        width = self.grid_size if W_patches is None else int(W_patches)
        if T != 1 or tokens != height * width:
            raise ValueError('PSP attention requires one complete latent frame')
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        ids = torch.arange(tokens, device=x.device)
        positions = (ids // (height * width), ids // width, ids % width)
        def rotate(value):
            pieces = [rotate_queries_or_keys(value[..., i*self.axis_dim:(i+1)*self.axis_dim], pos=pos)
                      for i, pos in enumerate(positions)]
            if 3 * self.axis_dim < self.head_dim:
                pieces.append(value[..., 3*self.axis_dim:])
            return torch.cat(pieces, dim=-1)
        q, k = rotate(q), rotate(k)
        if self.use_sdpa:
            value = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        else:
            value = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1) @ v
        return self.proj(value.transpose(1, 2).reshape(batch, tokens, dim))


class _WaveSpatialBlock(nn.Module):
    """Spatial Transformer block with single-frame RoPE."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0,
                 attn_drop=0.0, drop_path=0.0, use_sdpa=True, is_causal=False,
                 grid_size=16, use_rope=True):
        super().__init__()
        if drop or attn_drop or drop_path or is_causal or not use_rope:
            raise ValueError('Wave PSP uses noncausal RoPE attention without dropout')
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WaveRoPEAttention(dim, num_heads, qkv_bias, use_sdpa, grid_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x, *, T=1, H_patches=None, W_patches=None):
        x = x + self.attn(self.norm1(x), T=T, H_patches=H_patches, W_patches=W_patches)
        return x + self.mlp(self.norm2(x))


class _WaveFieldHead(nn.Module):
    def __init__(self, hidden_dim, dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
                                 nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, hidden):
        return self.net(hidden)


class Wave2DPhysicsStructuredPredictor(nn.Module):
    """Wave-2D: C(q) + (c²/s_c²) W(q) − (k/s_k) D(q) + closure_scale R(q,c,k).

    The residual term uses a conditional FiLM closure.
    Conditions contain raw [c, k], and the output is a latent time derivative.
    Default scales are ID-training RMS(c²) and RMS(k); recompute
    them when training on a different coefficient distribution.
    """

    def __init__(self, *, dim=192, hidden_dim=384, grid_size=(16, 16), depth=24,
                 num_heads=12, mlp_ratio=3.859375, closure_hidden_ratio=0.5,
                 closure_depth=2, scale_c2=124973.36530299438,
                 scale_k=28.64597816617425, closure_scale=1.0, use_rope=True,
                 use_sdpa=True, dropout=0.0, attention_dropout=0.0, qkv_bias=True,
                 activation_checkpointing=False, init_std=0.02, head_init_std=1e-3):
        super().__init__()
        hidden_dim = dim if hidden_dim is None else int(hidden_dim)
        if min(dim, hidden_dim, depth, num_heads, *grid_size, closure_depth) < 1 or hidden_dim % num_heads:
            raise ValueError('Invalid Wave PSP dimensions')
        if not use_rope or dropout or attention_dropout:
            raise ValueError('Wave PSP uses RoPE and zero dropout')
        if (not all(math.isfinite(float(v)) for v in (scale_c2, scale_k, closure_hidden_ratio, closure_scale))
                or min(scale_c2, scale_k, closure_hidden_ratio) <= 0):
            raise ValueError('Coefficient scales and closure width must be positive')
        self.dim, self.hidden_dim = int(dim), hidden_dim
        self.grid_size = tuple(int(v) for v in grid_size)
        self.num_tokens = self.grid_size[0] * self.grid_size[1]
        self.closure_scale = float(closure_scale)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.init_std = float(init_std)
        self.register_buffer('scale_c2', torch.tensor(float(scale_c2), dtype=torch.float32))
        self.register_buffer('scale_k', torch.tensor(float(scale_k), dtype=torch.float32))
        self.input_norm = nn.LayerNorm(self.dim)
        self.input_projection = nn.Identity() if hidden_dim == dim else nn.Linear(dim, hidden_dim)
        self.trunk = nn.ModuleList([
            _WaveSpatialBlock(hidden_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias,
                              grid_size=self.grid_size[0], use_sdpa=use_sdpa)
            for _ in range(depth)
        ])
        self.c_head, self.w_head, self.d_head = [_WaveFieldHead(hidden_dim, dim) for _ in range(3)]
        self.param_mlp = nn.Sequential(nn.Linear(2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2*hidden_dim))
        self.closure_norm = nn.LayerNorm(hidden_dim)
        width = max(1, round(hidden_dim * closure_hidden_ratio))
        layers, in_dim = [], hidden_dim
        for _ in range(closure_depth - 1):
            layers.extend([nn.Linear(in_dim, width), nn.GELU()])
            in_dim = width
        self.closure_mlp = nn.Sequential(*layers)
        self.closure_out = nn.Linear(in_dim, dim)
        self.apply(self._init_weights)
        for index, block in enumerate(self.trunk, 1):
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * index))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * index))
        for head in (self.c_head, self.w_head, self.d_head):
            trunc_normal_(head.net[-1].weight, std=head_init_std)
            nn.init.zeros_(head.net[-1].bias)
        nn.init.zeros_(self.closure_out.weight)
        nn.init.zeros_(self.closure_out.bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_hidden(self, q):
        if q.ndim != 3 or q.shape[1:] != (self.num_tokens, self.dim):
            raise ValueError(f'Expected latent [B,{self.num_tokens},{self.dim}]')
        if not bool(torch.isfinite(q).all()):
            raise ValueError('Latent state contains NaN or Inf')
        hidden = self.input_projection(self.input_norm(q))
        for block in self.trunk:
            def run(value, module=block):
                return module(value, T=1, H_patches=self.grid_size[0], W_patches=self.grid_size[1])
            hidden = checkpoint(run, hidden, use_reentrant=False) if (
                self.activation_checkpointing and self.training) else run(hidden)
        return hidden

    def forward(self, q, conditions, k=None):
        if k is not None:
            conditions = torch.stack((conditions.reshape(-1), k.reshape(-1)), dim=-1)
        if conditions.shape != (q.shape[0], 2):
            raise ValueError('Wave conditions must be raw [c,k] with shape [B,2]')
        if not bool(torch.isfinite(conditions).all()):
            raise ValueError('Wave conditions contain NaN or Inf')
        c, k = conditions.float().unbind(-1)
        rc2, rk = c.square() / self.scale_c2, k / self.scale_k
        hidden = self.forward_hidden(q)
        base, wave, damping = self.c_head(hidden), self.w_head(hidden), self.d_head(hidden)
        gamma, beta = self.param_mlp(torch.stack((rc2, rk), -1).to(hidden.dtype)).chunk(2, -1)
        closure_hidden = self.closure_norm(hidden) * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        closure = self.closure_out(self.closure_mlp(closure_hidden))
        return (base + rc2.to(base.dtype)[:, None, None] * wave
                - rk.to(base.dtype)[:, None, None] * damping + self.closure_scale * closure)


class _GrayScottFieldHead(nn.Module):
    def __init__(self, embed_dim, dim, hidden_dim, output_bias):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.hidden = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.activation = nn.GELU()
        self.output = nn.Linear(hidden_dim, dim, bias=output_bias)

    def forward(self, hidden):
        return self.output(self.activation(self.hidden(self.norm(hidden))))


class GrayScottPhysicsStructuredPredictor(nn.Module):
    """Gray–Scott: C(q) + (F/s_F) A_F(q) + (k/s_k) A_k(q).

    The spatial trunk and all three heads are independent of the parameters.
    Conditions contain raw [F, k]. Default scales are ID-training
    mean absolute values; recompute them when using different training data.
    """

    def __init__(self, *, dim=192, grid_size=(8, 8), embed_dim=384, depth=8,
                 num_heads=12, mlp_ratio=4.0, scale_F=0.03646408256298552,
                 scale_k=0.06144145619124174, qkv_bias=True, dropout=0.0,
                 attention_dropout=0.0, use_sdpa=True, activation_checkpointing=False,
                 init_std=0.02, head_init_std=1e-3, head_hidden_dim=85):
        super().__init__()
        if min(dim, embed_dim, depth, num_heads, head_hidden_dim, *grid_size) < 1:
            raise ValueError('Gray–Scott PSP dimensions must be positive')
        if (embed_dim % num_heads or embed_dim % 4 or min(scale_F, scale_k) <= 0
                or not all(math.isfinite(float(v)) for v in (scale_F, scale_k))):
            raise ValueError('Invalid hidden dimension or coefficient scales')
        if dropout or attention_dropout:
            raise ValueError('Gray–Scott PSP uses zero dropout')
        self.dim, self.grid_size = int(dim), tuple(int(v) for v in grid_size)
        self.num_tokens = self.grid_size[0] * self.grid_size[1]
        self.activation_checkpointing, self.init_std = bool(activation_checkpointing), float(init_std)
        self.register_buffer('scale_F', torch.tensor(float(scale_F), dtype=torch.float32))
        self.register_buffer('scale_k', torch.tensor(float(scale_k), dtype=torch.float32))
        self.input_norm = nn.LayerNorm(dim)
        self.input_projection = nn.Linear(dim, embed_dim)
        grid_w, grid_h = np.meshgrid(np.arange(self.grid_size[1], dtype=np.float64),
                                      np.arange(self.grid_size[0], dtype=np.float64))
        position = np.concatenate([get_1d_sincos_pos_embed_from_grid(embed_dim//2, grid_h),
                                   get_1d_sincos_pos_embed_from_grid(embed_dim//2, grid_w)], axis=1)
        self.register_buffer('spatial_positional_embedding', torch.from_numpy(position.astype(np.float32)).unsqueeze(0))
        self.trunk = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias,
                                          use_sdpa=use_sdpa) for _ in range(depth)])
        self.base_head = _GrayScottFieldHead(embed_dim, dim, head_hidden_dim, True)
        self.feed_head = _GrayScottFieldHead(embed_dim, dim, head_hidden_dim, False)
        self.kill_head = _GrayScottFieldHead(embed_dim, dim, head_hidden_dim, False)
        self.apply(self._init_weights)
        for index, block in enumerate(self.trunk, 1):
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * index))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * index))
        for head in (self.base_head, self.feed_head, self.kill_head):
            trunc_normal_(head.output.weight, std=head_init_std)
            if head.output.bias is not None:
                nn.init.zeros_(head.output.bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_hidden(self, q):
        if q.ndim != 3 or q.shape[1:] != (self.num_tokens, self.dim):
            raise ValueError(f'Expected latent [B,{self.num_tokens},{self.dim}]')
        if not bool(torch.isfinite(q).all()):
            raise ValueError('Latent state contains NaN or Inf')
        hidden = self.input_projection(self.input_norm(q))
        hidden = hidden + self.spatial_positional_embedding.to(hidden.dtype)
        for block in self.trunk:
            hidden = checkpoint(block, hidden, use_reentrant=False) if (
                self.activation_checkpointing and self.training) else block(hidden)
        return hidden

    def forward_fields(self, q):
        hidden = self.forward_hidden(q)
        return self.base_head(hidden), self.feed_head(hidden), self.kill_head(hidden)

    def forward(self, q, conditions, k=None):
        if k is not None:
            conditions = torch.stack((conditions.reshape(-1), k.reshape(-1)), dim=-1)
        if conditions.shape != (q.shape[0], 2):
            raise ValueError('Gray–Scott conditions must be raw [F,k] with shape [B,2]')
        if not bool(torch.isfinite(conditions).all()):
            raise ValueError('Gray–Scott conditions contain NaN or Inf')
        feed, kill = conditions.to(device=q.device, dtype=torch.float32).unbind(-1)
        base, feed_field, kill_field = self.forward_fields(q)
        r_feed, r_kill = feed / self.scale_F, kill / self.scale_k
        return (base + r_feed.to(feed_field.dtype)[:, None, None] * feed_field
                + r_kill.to(kill_field.dtype)[:, None, None] * kill_field)


def _sincos_1d(length: int, dim: int) -> torch.Tensor:
    if length < 1 or dim < 4 or dim % 2:
        raise ValueError("1-D sin/cos embedding requires length>0 and positive even dim>=4")
    grid = np.arange(length, dtype=np.float64)
    values = get_1d_sincos_pos_embed_from_grid(dim, grid)
    return torch.from_numpy(values.astype(np.float32))


class _SpatialTransformer1D(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 192,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 12,
        num_tokens: int = 32,
        use_sdpa: bool = True,
        head_init_std: float = 1.0e-3,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.num_tokens = int(num_tokens)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.register_buffer("position", _sincos_1d(num_tokens, embed_dim).unsqueeze(0), persistent=True)
        self.blocks = nn.ModuleList(
            [
                Block(
                    use_rope=False,
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                    use_sdpa=use_sdpa,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, input_dim)
        self.apply(self._init_weights)
        self._rescale_blocks()
        trunc_normal_(self.output_projection.weight, std=float(head_init_std))
        nn.init.zeros_(self.output_projection.bias)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _rescale_blocks(self) -> None:
        for index, block in enumerate(self.blocks, start=1):
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * index))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * index))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        if q.ndim != 3 or q.shape[1:] != (self.num_tokens, self.input_dim):
            raise ValueError(f"Expected [B,{self.num_tokens},{self.input_dim}], received {tuple(q.shape)}")
        hidden = self.input_projection(self.input_norm(q))
        hidden = hidden + self.position.to(dtype=hidden.dtype)
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        return self.output_projection(self.output_norm(hidden))


class AdvectionPhysicsStructuredPredictor(nn.Module):
    """Zero-intercept dq/dt=(beta/s_beta) A_theta(q)."""

    def __init__(self, *, beta_scale: float, **kwargs) -> None:
        super().__init__()
        if not math.isfinite(float(beta_scale)) or float(beta_scale) <= 0.0:
            raise ValueError("beta_scale must be finite and positive")
        self.beta_scale = float(beta_scale)
        self.unit_advection = _SpatialTransformer1D(**kwargs)

    def unit_field(self, q: torch.Tensor) -> torch.Tensor:
        return self.unit_advection(q)

    def forward(self, q: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        if beta.ndim == 1:
            beta = beta.unsqueeze(-1)
        return (beta.float() / self.beta_scale).unsqueeze(-1) * self.unit_field(q)




class WaveBoundaryPhysicsStructuredPredictor(nn.Module):
    """Non-periodic, boundary-conditioned second-order Wave-B predictor."""

    def __init__(
        self,
        *,
        latent_dim: int = 192,
        hidden_dim: int = 384,
        num_tokens: int = 64,
        depth: int = 24,
        num_heads: int = 12,
        mlp_ratio: float = 3.9010416666666665,
        velocity_scale: float,
        wave_speed: float = 2.0,
        scale_c2: float = 4.0,
        condition_dim: int = 4,
        closure_hidden_ratio: float = 0.5,
        closure_depth: int = 2,
        closure_scale: float = 1.0,
        use_rope: bool = True,
        use_sdpa: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        qkv_bias: bool = True,
        activation_checkpointing: bool = False,
        init_std: float = 0.02,
        head_init_std: float = 1.0e-3,
    ) -> None:
        super().__init__()
        values = (latent_dim, hidden_dim, num_tokens, depth, num_heads, condition_dim)
        if any(int(value) < 1 for value in values):
            raise ValueError("model dimensions must be positive")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not math.isfinite(float(velocity_scale)) or velocity_scale <= 0.0:
            raise ValueError("velocity_scale must be finite and positive")
        if not math.isfinite(float(wave_speed)) or wave_speed <= 0.0:
            raise ValueError("wave_speed must be finite and positive")
        if not math.isfinite(float(scale_c2)) or scale_c2 <= 0.0:
            raise ValueError("scale_c2 must be finite and positive")
        if closure_depth < 1 or closure_hidden_ratio <= 0.0:
            raise ValueError("closure settings must be positive")
        if dropout != 0.0 or attention_dropout != 0.0:
            raise ValueError("Wave-B structured predictor requires zero dropout")
        rotary_axis_dim = 2 * (((hidden_dim // num_heads) // 3) // 2)
        if not use_rope:
            raise ValueError("Wave-B requires a nonperiodic RoPE trunk")
        if rotary_axis_dim < 4:
            raise ValueError("RoPE requires at least four features per axis")

        self.latent_dim = int(latent_dim)
        self.state_dim = 2 * self.latent_dim
        self.hidden_dim = int(hidden_dim)
        self.num_tokens = int(num_tokens)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.condition_dim = int(condition_dim)
        self.closure_hidden_ratio = float(closure_hidden_ratio)
        self.closure_depth = int(closure_depth)
        self.closure_scale = float(closure_scale)
        self.use_rope = bool(use_rope)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.init_std = float(init_std)
        self.head_init_std = float(head_init_std)

        self.register_buffer(
            "velocity_scale", torch.tensor(float(velocity_scale), dtype=torch.float32)
        )
        self.register_buffer(
            "wave_speed", torch.tensor(float(wave_speed), dtype=torch.float32)
        )
        self.register_buffer(
            "scale_c2", torch.tensor(float(scale_c2), dtype=torch.float32)
        )

        self.input_norm = nn.LayerNorm(self.state_dim)
        self.input_projection = (
            nn.Identity()
            if self.state_dim == self.hidden_dim
            else nn.Linear(self.state_dim, self.hidden_dim)
        )
        self.trunk = nn.ModuleList(
            [
                _WaveSpatialBlock(
                    dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=bool(qkv_bias),
                    drop=float(dropout),
                    attn_drop=float(attention_dropout),
                    drop_path=0.0,
                    use_sdpa=bool(use_sdpa),
                    is_causal=False,
                    grid_size=self.num_tokens,
                    use_rope=self.use_rope,
                )
                for _ in range(self.depth)
            ]
        )
        self.c_head = _WaveFieldHead(self.hidden_dim, self.latent_dim)
        self.w_head = _WaveFieldHead(self.hidden_dim, self.latent_dim)

        self.condition_mlp = nn.Sequential(
            nn.Linear(self.condition_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
        )
        self.closure_norm = nn.LayerNorm(self.hidden_dim)
        closure_hidden = max(
            1, int(round(self.hidden_dim * self.closure_hidden_ratio))
        )
        closure_layers: list[nn.Module] = []
        closure_input = self.hidden_dim
        for _ in range(self.closure_depth - 1):
            closure_layers.extend(
                [nn.Linear(closure_input, closure_hidden), nn.GELU()]
            )
            closure_input = closure_hidden
        self.closure_mlp = nn.Sequential(*closure_layers)
        self.closure_out = nn.Linear(closure_input, self.latent_dim)

        self.apply(self._init_weights)
        self._rescale_trunk()
        for head in (self.c_head, self.w_head):
            output = head.net[-1]
            assert isinstance(output, nn.Linear)
            trunc_normal_(output.weight, std=self.head_init_std)
            nn.init.zeros_(output.bias)
        # At initialization the boundary-conditioned intervention is absent.
        nn.init.zeros_(self.closure_out.weight)
        nn.init.zeros_(self.closure_out.bias)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _rescale_trunk(self) -> None:
        for layer_index, block in enumerate(self.trunk, start=1):
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * layer_index))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * layer_index))

    def _validate_inputs(
        self, state: torch.Tensor, boundary: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.ndim != 3 or state.shape[1:] != (
            self.num_tokens,
            self.state_dim,
        ):
            raise ValueError(
                "state must be [B,N,2D] with trailing shape "
                f"{(self.num_tokens, self.state_dim)}, got {tuple(state.shape)}"
            )
        if boundary.ndim != 2 or boundary.shape != (
            state.shape[0],
            self.condition_dim,
        ):
            raise ValueError(
                f"boundary must be [B,{self.condition_dim}], got {tuple(boundary.shape)}"
            )
        if not bool(torch.isfinite(state).all()) or not bool(
            torch.isfinite(boundary).all()
        ):
            raise ValueError("state or boundary contains NaN/Inf")
        return state, boundary.float()

    def forward_hidden(self, state: torch.Tensor) -> torch.Tensor:
        """Evaluate the boundary-independent non-periodic spatial trunk."""

        hidden = self.input_projection(self.input_norm(state))
        for block in self.trunk:
            if self.activation_checkpointing and self.training:
                hidden = torch.utils.checkpoint.checkpoint(
                    lambda value, module=block: module(
                        value,
                        T=1,
                        H_patches=self.num_tokens,
                        W_patches=1,
                    ),
                    hidden,
                    use_reentrant=False,
                )
            else:
                hidden = block(
                    hidden,
                    T=1,
                    H_patches=self.num_tokens,
                    W_patches=1,
                )
        return hidden

    def forward(
        self,
        state: torch.Tensor,
        boundary: torch.Tensor,
    ):
        state, boundary = self._validate_inputs(state, boundary)
        q, velocity = state.split(self.latent_dim, dim=-1)
        del q  # q remains present inside ``state`` consumed by the trunk.
        hidden = self.forward_hidden(state)
        conservative = self.c_head(hidden)
        wave = self.w_head(hidden)

        gamma, beta = self.condition_mlp(
            boundary.to(dtype=hidden.dtype)
        ).chunk(2, dim=-1)
        closure_hidden = self.closure_norm(hidden)
        closure_hidden = (
            closure_hidden * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        )
        closure = self.closure_out(self.closure_mlp(closure_hidden))

        wave_coefficient = self.wave_speed.square() / self.scale_c2
        acceleration = (
            conservative
            + wave_coefficient.to(dtype=wave.dtype) * wave
            + self.closure_scale * closure
        )
        q_dot = self.velocity_scale.to(dtype=velocity.dtype) * velocity
        derivative = torch.cat((q_dot, acceleration), dim=-1)
        return derivative


FORCED_CONDITION_DIM = 7


FORCED_CONDITION_SCHEME = "log_beta_plus_interval_mean_fourier_l1_l2_l3_v1"


FORCED_CONDITION_ORDER = ("log_beta", "sin_l1", "cos_l1", "sin_l2", "cos_l2", "sin_l3", "cos_l3")


class ForcedPhysicsConditionedPredictor(nn.Module):
    """Conditional Markov transition for Burgers/Heat.

    ``forward(q, condition)`` returns the next state; ``predict_delta`` exposes
    the learned increment. Every call observes exactly one state.
    """

    transition_mode = "markov_one_step"
    output_mode = "next_state"
    is_continuous = False
    context_frames = 1
    condition_dim = FORCED_CONDITION_DIM
    condition_input_type = FORCED_CONDITION_SCHEME

    def __init__(self, *, spatial_points=256, patch_size=4, embed_dim=192,
                 predictor_embed_dim=384, depth=24, num_heads=12,
                 use_activation_checkpointing=True, use_sdpa=True,
                 zero_init_output=True, init_std=0.02):
        super().__init__()
        if min(spatial_points, patch_size, embed_dim, predictor_embed_dim, depth, num_heads) < 1:
            raise ValueError("Model dimensions must be positive")
        if spatial_points % patch_size or predictor_embed_dim % num_heads:
            raise ValueError("Invalid patch grid or attention head width")
        self.grid_height, self.grid_width = int(spatial_points) // int(patch_size), 1
        self.input_embed_dim = int(embed_dim)
        self.num_frames = 1
        self.use_activation_checkpointing = bool(use_activation_checkpointing)
        self.init_std = float(init_std)
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.action_encoder = nn.Linear(FORCED_CONDITION_DIM, predictor_embed_dim)
        self.predictor_blocks = nn.ModuleList([
            ACBlock(predictor_embed_dim, num_heads=num_heads, norm_layer=norm,
                    grid_size=self.grid_height, use_sdpa=use_sdpa)
            for _ in range(depth)
        ])
        self.predictor_norm = norm(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim)
        self.apply(self._init_weights)
        for index, block in enumerate(self.predictor_blocks, start=1):
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * index))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * index))
        if zero_init_output:
            nn.init.zeros_(self.predictor_proj.weight)
            nn.init.zeros_(self.predictor_proj.bias)
        self.register_buffer("attn_mask", build_action_block_causal_attention_mask(
            1, self.grid_height, 1, add_tokens=1), persistent=False)

    def _init_weights(self, module):
        AuxiliaryDynamicsPredictor._init_weights(self, module)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load weights strictly, excluding unused extrinsics-encoder entries."""
        if not strict:
            raise ValueError("Forced predictor restoration requires strict=True")
        ignored_keys = ("extrinsics_encoder.weight", "extrinsics_encoder.bias")
        present = tuple(key for key in ignored_keys if key in state_dict)
        if present and len(present) != 2:
            raise ValueError("Extrinsics-encoder entries must include both weight and bias")
        filtered = OrderedDict((key, value) for key, value in state_dict.items() if key not in ignored_keys)
        if hasattr(state_dict, "_metadata"):
            filtered._metadata = state_dict._metadata
        return super().load_state_dict(filtered, strict=True, assign=assign)

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        """Construct from a checkpoint's predictor configuration."""
        p = config["predictor"]
        if int(p.get("action_embed_dim", 7)) != 7:
            raise ValueError("Forced Burgers/Heat require seven condition coordinates")
        if not p.get("pred_is_frame_causal", True) or not p.get("use_rope", True) or p.get("use_pred_silu", False):
            raise ValueError("Expected causal RoPE attention with GELU activation")
        if p.get("transition_mode", "markov_one_step") != "markov_one_step":
            raise ValueError("Only a one-state Markov transition is supported")
        return cls(spatial_points=int(config["data"]["spatial_points"]),
                   patch_size=int(config["data"]["patch_size"]),
                   embed_dim=int(config["encoder"]["embed_dim"]),
                   predictor_embed_dim=int(p["pred_embed_dim"]), depth=int(p["pred_depth"]),
                   num_heads=int(p["pred_num_heads"]),
                   use_activation_checkpointing=bool(p.get("use_activation_checkpointing", True)),
                   use_sdpa=bool(config.get("meta", {}).get("use_sdpa", True)),
                   zero_init_output=bool(p.get("zero_init_delta_projection", True)))

    def predict_delta(self, q: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        expected = (self.grid_height, self.input_embed_dim)
        if q.ndim != 3 or tuple(q.shape[1:]) != expected:
            raise ValueError(f"Expected q [B,{expected[0]},{expected[1]}], got {tuple(q.shape)}")
        if condition.shape != (q.shape[0], FORCED_CONDITION_DIM):
            raise ValueError(f"Expected condition [B,7], got {tuple(condition.shape)}")
        x = self.predictor_embed(q)
        batch, tokens, width = x.shape
        action = self.action_encoder(condition[:, None]).unsqueeze(2)
        x = torch.cat((action, x.view(batch, 1, tokens, width)), dim=2).flatten(1, 2)
        mask = self.attn_mask.to(x.device, non_blocking=True)
        for block in self.predictor_blocks:
            kwargs = dict(mask=None, attn_mask=mask, T=1, H=self.grid_height, W=1, action_tokens=1)
            if self.training and self.use_activation_checkpointing:
                x = checkpoint(block, x, use_reentrant=False, **kwargs)
            else:
                x = block(x, **kwargs)
        x = x.view(batch, 1, 1 + tokens, width)[:, :, 1:].flatten(1, 2)
        return self.predictor_proj(self.predictor_norm(x))

    def forward(self, q: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return q + self.predict_delta(q, condition)


def interval_fourier_coefficients(amplitude, omega, harmonic, phase, physical_time):
    """Interval-average [sin_l1,cos_l1,sin_l2,cos_l2,sin_l3,cos_l3] coefficients."""
    if amplitude.ndim != 2 or amplitude.shape[1] != 5:
        raise ValueError("Forcing amplitudes must be [B,5]")
    if any(x.shape != amplitude.shape for x in (omega, harmonic, phase)):
        raise ValueError("All forcing mode arrays must have the same [B,5] shape")
    if physical_time.ndim == 1:
        physical_time = physical_time.unsqueeze(0).expand(amplitude.shape[0], -1)
    if physical_time.ndim != 2 or physical_time.shape[0] != amplitude.shape[0] or physical_time.shape[1] < 2:
        raise ValueError("Physical times must be [T] or [B,T]")
    delta_t = physical_time[:, 1:] - physical_time[:, :-1]
    if not bool(torch.isfinite(delta_t).all()) or bool((delta_t <= 0).any()):
        raise ValueError("Physical times must be finite and strictly increasing")
    midpoint = 0.5 * (physical_time[:, :-1] + physical_time[:, 1:])
    angle = omega[:, None, :] * midpoint[:, :, None] + phase[:, None, :]
    interval_scale = torch.sinc(omega[:, None, :] * delta_t[:, :, None] / (2.0 * math.pi))
    weighted = amplitude[:, None, :] * interval_scale
    sine_coeff = weighted * torch.cos(angle)
    cosine_coeff = weighted * torch.sin(angle)
    columns = []
    for mode in (1, 2, 3):
        selector = (harmonic == mode)[:, None, :]
        columns.append(torch.sum(sine_coeff * selector, dim=-1))
        columns.append(torch.sum(cosine_coeff * selector, dim=-1))
    result = torch.stack(columns, dim=-1)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Nonfinite interval Fourier coefficients")
    return result


def build_forced_transition_condition(beta, amplitude, omega, harmonic, phase,
                                      physical_time, stats: dict[str, Any]):
    """Construct [B,T-1,7] using frozen, training-only checkpoint statistics.

    Required stats: log_beta_mean/std and six-entry forcing_mean/std from
    ``condition_authority``. These statistics are fixed during training.
    """
    beta = beta.float().reshape(-1)
    if not bool(torch.isfinite(beta).all()) or bool((beta <= 0).any()):
        raise ValueError("Beta must be finite and positive")
    coefficients = interval_fourier_coefficients(amplitude.float(), omega.float(),
                                                harmonic.long(), phase.float(), physical_time.float())
    beta_value = (torch.log(beta) - float(stats["log_beta_mean"])) / float(stats["log_beta_std"])
    mean = torch.as_tensor(stats["forcing_mean"], device=coefficients.device, dtype=coefficients.dtype)
    std = torch.as_tensor(stats["forcing_std"], device=coefficients.device, dtype=coefficients.dtype)
    forcing_value = (coefficients - mean) / std
    result = torch.cat((beta_value[:, None, None].expand(-1, coefficients.shape[1], 1), forcing_value), dim=-1)
    if result.shape != (beta.shape[0], coefficients.shape[1], FORCED_CONDITION_DIM):
        raise RuntimeError("Unexpected forced-condition shape")
    return result


class CombinedPhysicsStructuredPredictor(nn.Module):
    def __init__(self, *, scale_alpha, scale_beta, scale_gamma, dim=192,
                 hidden_dim=384, depth=24, num_heads=12, num_tokens=64,
                 mlp_ratio=4.0, use_sdpa=True, activation_checkpointing=False,
                 init_std=0.02, head_init_std=1e-3):
        super().__init__()
        if min(dim, hidden_dim, depth, num_heads, num_tokens) < 1 or hidden_dim % num_heads:
            raise ValueError('Invalid predictor dimensions')
        scales = (scale_alpha, scale_beta, scale_gamma)
        if any(not math.isfinite(float(value)) or value <= 0 for value in scales):
            raise ValueError('ID-training RMS coefficient scales must be finite and positive')
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0:
            raise ValueError('MLP ratio must be finite and positive')
        self.dim, self.num_tokens = int(dim), int(num_tokens)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.init_std = float(init_std)
        for name, value in zip(('scale_alpha', 'scale_beta', 'scale_gamma'), scales):
            self.register_buffer(name, torch.tensor(float(value), dtype=torch.float32))
        self.input_norm = nn.LayerNorm(dim)
        self.input_projection = nn.Linear(dim, hidden_dim)
        self.trunk = nn.ModuleList([
            _WaveSpatialBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio,
                              qkv_bias=True, grid_size=num_tokens, use_sdpa=use_sdpa)
            for _ in range(depth)
        ])
        self.alpha_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, dim))
        self.beta_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, dim))
        self.gamma_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, dim))
        self.apply(self._init_weights)
        for index, block in enumerate(self.trunk, start=1):
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * index))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * index))
        for head in (self.alpha_head, self.beta_head, self.gamma_head):
            trunc_normal_(head[-1].weight, std=head_init_std)
            nn.init.zeros_(head[-1].bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_fields(self, q):
        if q.ndim != 3 or q.shape[1:] != (self.num_tokens, self.dim):
            raise ValueError(f'Expected latent [B,{self.num_tokens},{self.dim}]')
        if not bool(torch.isfinite(q).all()):
            raise ValueError('Latent state contains NaN or Inf')
        hidden = self.input_projection(self.input_norm(q))
        for block in self.trunk:
            def run(value, module=block):
                return module(value, T=1, H_patches=self.num_tokens, W_patches=1)
            hidden = checkpoint(run, hidden, use_reentrant=False) if (
                self.activation_checkpointing and self.training) else run(hidden)
        return self.alpha_head(hidden), self.beta_head(hidden), self.gamma_head(hidden)

    def forward(self, q, conditions):
        if conditions.shape != (q.shape[0], 3):
            raise ValueError('Conditions must be raw [alpha,beta,gamma] with shape [B,3]')
        if not bool(torch.isfinite(conditions).all()):
            raise ValueError('Conditions contain NaN or Inf')
        alpha, beta, gamma = conditions.to(device=q.device, dtype=torch.float32).unbind(-1)
        alpha_field, beta_field, gamma_field = self.forward_fields(q)
        r_alpha = (alpha / self.scale_alpha).to(alpha_field.dtype)[:, None, None]
        r_beta = (beta / self.scale_beta).to(beta_field.dtype)[:, None, None]
        r_gamma = (gamma / self.scale_gamma).to(gamma_field.dtype)[:, None, None]
        return -r_alpha * alpha_field + r_beta * beta_field - r_gamma * gamma_field


_PSP_BUILDERS = {
    "vorticity": VorticityPhysicsStructuredPredictor,
    "advection": AdvectionPhysicsStructuredPredictor,
    "wave2d": Wave2DPhysicsStructuredPredictor,
    "grayscott": GrayScottPhysicsStructuredPredictor,
    "wave_b": WaveBoundaryPhysicsStructuredPredictor,
    "burgers": ForcedPhysicsConditionedPredictor,
    "heat": ForcedPhysicsConditionedPredictor,
    "combined": CombinedPhysicsStructuredPredictor,
}


def build_psp(task="vorticity", **kwargs):
    """Build a task predictor; keyword arguments override its defaults.

    Vector-field models return dq/dt for ``fixed_step_rk4``. Burgers/Heat
    return q_next directly.
    Advection requires beta_scale; Wave-B requires velocity_scale. These
    statistics must come from the ID training split or its saved checkpoint.
    """
    name = str(task).lower().replace("-", "_")
    name = {"wave_2d": "wave2d", "gray_scott": "grayscott"}.get(name, name)
    if name not in _PSP_BUILDERS:
        raise ValueError(f"Unknown PSP task {task!r}; available: {', '.join(_PSP_BUILDERS)}")
    return _PSP_BUILDERS[name](**kwargs)
