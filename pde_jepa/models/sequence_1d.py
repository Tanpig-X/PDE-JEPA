# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license in the repository LICENSE file.
"""One-dimensional field encoders and framewise decoders."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from pde_jepa.models.spatial_blocks import MLP, rotate_queries_or_keys
from pde_jepa.utils.tensors import trunc_normal_

class _SequenceRoPEAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, use_sdpa=True, grid_size=14, is_causal=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.d_dim = int(2 * (head_dim // 3 // 2))
        self.h_dim = int(2 * (head_dim // 3 // 2))
        self.w_dim = int(2 * (head_dim // 3 // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal

    def _get_frame_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
        else:
            tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        width_ids = ids - tokens_per_frame * frame_ids - tokens_per_row * height_ids
        return (frame_ids, height_ids, width_ids)

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        B, N, C = x.size()
        grid_depth = int(N // (self.grid_size * self.grid_size))
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)
        else:
            if T is None or H_patches is None or W_patches is None:
                mask = torch.arange(int(grid_depth * self.grid_size * self.grid_size), device=x.device)
            else:
                mask = torch.arange(int(T * H_patches * W_patches), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)
        s = 0
        qd = rotate_queries_or_keys(q[..., s:s + self.d_dim], pos=d_mask)
        kd = rotate_queries_or_keys(k[..., s:s + self.d_dim], pos=d_mask)
        s += self.d_dim
        qh = rotate_queries_or_keys(q[..., s:s + self.h_dim], pos=h_mask)
        kh = rotate_queries_or_keys(k[..., s:s + self.h_dim], pos=h_mask)
        s += self.h_dim
        qw = rotate_queries_or_keys(q[..., s:s + self.w_dim], pos=w_mask)
        kw = rotate_queries_or_keys(k[..., s:s + self.w_dim], pos=w_mask)
        s += self.w_dim
        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)
        if attn_mask is not None or self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask)
                attn = None
        else:
            attn = q @ k.transpose(-2, -1) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class _SequenceBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0, use_sdpa=True, grid_size=16, use_rope=True):
        super().__init__()
        if drop or attn_drop or drop_path or (not use_rope):
            raise ValueError('Sequence encoder requires RoPE and zero dropout')
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _SequenceRoPEAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, use_sdpa=use_sdpa, grid_size=grid_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        x = x + self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)
        return x + self.mlp(self.norm2(x))

def _gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.ndim != 2 or indices.shape[0] != tokens.shape[0]:
        raise ValueError('indices must be [B,K] and match token batch')
    return torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))

class PatchEmbed1D(nn.Module):

    def __init__(self, in_channels: int=1, embed_dim: int=192, patch_length: int=8, tubelet_size: int=1) -> None:
        super().__init__()
        self.patch_length = int(patch_length)
        self.tubelet_size = int(tubelet_size)
        if self.patch_length < 1 or self.tubelet_size < 1:
            raise ValueError('patch_length and tubelet_size must be positive')
        self.proj = nn.Conv3d(int(in_channels), int(embed_dim), kernel_size=(self.tubelet_size, 1, self.patch_length), stride=(self.tubelet_size, 1, self.patch_length))

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 5 or field.shape[-2] != 1:
            raise ValueError(f'Expected [B,C,T,1,X], received {tuple(field.shape)}')
        if field.shape[2] % self.tubelet_size:
            raise ValueError('temporal length must be divisible by tubelet_size')
        if field.shape[-1] % self.patch_length:
            raise ValueError('spatial length must be divisible by patch_length')
        return self.proj(field).flatten(2).transpose(1, 2)

class VJEPAEncoder1D(nn.Module):

    def __init__(self, *, spatial_points: int=256, patch_length: int=8, tubelet_size: int=1, in_channels: int=1, embed_dim: int=192, depth: int=12, num_heads: int=3, use_sdpa: bool=True, activation_checkpointing: bool=False, init_std: float=0.02) -> None:
        super().__init__()
        if spatial_points % patch_length:
            raise ValueError('spatial_points must be divisible by patch_length')
        self.spatial_points = int(spatial_points)
        self.patch_length = int(patch_length)
        self.tubelet_size = int(tubelet_size)
        if self.tubelet_size < 1:
            raise ValueError('tubelet_size must be positive')
        self.tokens_per_frame = self.spatial_points // self.patch_length
        self.grid_height = 1
        self.grid_width = self.tokens_per_frame
        self.hierarchical_layers = [int(depth) - 1]
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.patch_embed = PatchEmbed1D(in_channels, embed_dim, patch_length, tubelet_size=self.tubelet_size)
        self.blocks = nn.ModuleList([_SequenceBlock(use_rope=True, grid_size=self.tokens_per_frame, dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0, use_sdpa=use_sdpa) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.init_std = float(init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _rescale_blocks(self) -> None:
        for index, block in enumerate(self.blocks, start=1):
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * index))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * index))

    @property
    def use_activation_checkpointing(self):
        return self.activation_checkpointing

    @use_activation_checkpointing.setter
    def use_activation_checkpointing(self, value):
        self.activation_checkpointing = bool(value)

    def forward(self, field: torch.Tensor, mask: torch.Tensor | None=None,
                *, masks=None, training=False) -> torch.Tensor:
        if masks is not None:
            if mask is not None:
                raise ValueError("Specify either mask or masks")
            mask = masks
        input_frames = int(field.shape[2])
        if input_frames % self.tubelet_size:
            raise ValueError(f'input frames {input_frames} are not divisible by tubelet_size {self.tubelet_size}')
        runtime_frames = input_frames // self.tubelet_size
        tokens = self.patch_embed(field)
        if isinstance(mask, (list, tuple)):
            if not mask:
                raise ValueError("At least one token mask is required")
            tokens = torch.cat([_gather_tokens(tokens, item) for item in mask], dim=0)
            positions = torch.cat(list(mask), dim=0)
        elif mask is not None:
            tokens = _gather_tokens(tokens, mask)
            positions = mask
        else:
            positions = torch.arange(tokens.shape[1], device=tokens.device, dtype=torch.long).unsqueeze(0).expand(tokens.shape[0], -1)
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                tokens = torch.utils.checkpoint.checkpoint(block, tokens, positions, None, T=runtime_frames, H_patches=1, W_patches=self.tokens_per_frame, use_reentrant=False)
            else:
                tokens = block(tokens, mask=positions, T=runtime_frames, H_patches=1, W_patches=self.tokens_per_frame)
        return self.norm(tokens)

    def encode_frames(self, normalized_frames: torch.Tensor) -> torch.Tensor:
        if self.tubelet_size != 1:
            raise RuntimeError('encode_frames is only defined for tubelet_size=1; a temporal-tube encoder must receive complete tubes through forward()')
        if normalized_frames.ndim == 2:
            field = normalized_frames[:, None, None, None, :]
        elif normalized_frames.ndim == 4 and normalized_frames.shape[1:3] == (1, 1):
            field = normalized_frames.unsqueeze(2)
        else:
            raise ValueError(f'Expected [B,X] or [B,1,1,X], got {tuple(normalized_frames.shape)}')
        return self.forward(field)

class ProgressivePDEDecoder1D(nn.Module):

    def __init__(self, latent_dim: int=192, tokens: int=32, output_points: int=256, output_frames: int=1, spatial_padding_mode: str='circular', conv_channels: tuple[int, int, int, int] | list[int] | None=None) -> None:
        super().__init__()
        if tokens < 1 or output_points < tokens or output_points % tokens:
            raise ValueError('decoder output_points must be an integer multiple of tokens')
        if int(output_frames) < 1:
            raise ValueError('decoder output_frames must be positive')
        if spatial_padding_mode not in ('circular', 'reflect', 'replicate', 'zeros'):
            raise ValueError('unsupported one-dimensional decoder padding mode')
        ratio = output_points // tokens
        if ratio & ratio - 1 or ratio > 8:
            raise ValueError('decoder supports power-of-two upsampling ratios up to x8')
        self.latent_dim = int(latent_dim)
        self.tokens = int(tokens)
        self.output_points = int(output_points)
        self.output_frames = int(output_frames)
        self.spatial_padding_mode = str(spatial_padding_mode)
        self.upsampling_stages = int(math.log2(ratio))
        if conv_channels is None:
            conv_channels = (128, 96, 64, 32)
        self.conv_channels = tuple((int(value) for value in conv_channels))
        if len(self.conv_channels) != 4:
            raise ValueError('ProgressivePDEDecoder1D requires exactly four conv channels')
        if any((value < 1 for value in self.conv_channels)):
            raise ValueError('all ProgressivePDEDecoder1D conv channels must be positive')
        channel1, channel2, channel3, channel4 = self.conv_channels
        self.conv1 = nn.Conv1d(latent_dim, channel1, 3, padding=1, padding_mode=self.spatial_padding_mode)
        self.conv2 = nn.Conv1d(channel1, channel2, 3, padding=1, padding_mode=self.spatial_padding_mode)
        self.conv3 = nn.Conv1d(channel2, channel3, 3, padding=1, padding_mode=self.spatial_padding_mode)
        self.conv4 = nn.Conv1d(channel3, channel4, 3, padding=1, padding_mode=self.spatial_padding_mode)
        self.output = nn.Conv1d(channel4, self.output_frames, 3, padding=1, padding_mode=self.spatial_padding_mode)

    @staticmethod
    def _upsample(value: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value, scale_factor=2.0, mode='linear', align_corners=False)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        if q.ndim != 3 or q.shape[1:] != (self.tokens, self.latent_dim):
            raise ValueError(f'Expected [B,{self.tokens},{self.latent_dim}], received {tuple(q.shape)}')
        value = q.transpose(1, 2)
        value = F.gelu(self.conv1(value))
        if self.upsampling_stages >= 1:
            value = self._upsample(value)
        value = F.gelu(self.conv2(value))
        if self.upsampling_stages >= 2:
            value = self._upsample(value)
        value = F.gelu(self.conv3(value))
        if self.upsampling_stages >= 3:
            value = self._upsample(value)
        value = F.gelu(self.conv4(value))
        result = self.output(value)
        if result.shape[-1] != self.output_points:
            raise RuntimeError('decoder produced an unexpected spatial resolution')
        return result[:, 0] if self.output_frames == 1 else result

def _decoder_group_count(channels: int, maximum_groups: int) -> int:
    maximum_groups = min(int(maximum_groups), int(channels))
    for groups in range(maximum_groups, 0, -1):
        if channels % groups == 0:
            return groups
    return 1

def _decoder_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == 'gelu':
        return nn.GELU()
    if name == 'silu':
        return nn.SiLU()
    raise ValueError(f'decoder activation must be gelu or silu, received {name!r}')

class PeriodicResidualBlock1D(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, *, activation: str='gelu', norm_groups: int=8, spatial_padding_mode: str='circular') -> None:
        super().__init__()
        if spatial_padding_mode not in ('circular', 'reflect', 'replicate', 'zeros'):
            raise ValueError('unsupported one-dimensional decoder padding mode')
        groups = _decoder_group_count(int(out_channels), int(norm_groups))
        self.conv1 = nn.Conv1d(int(in_channels), int(out_channels), kernel_size=3, padding=1, padding_mode=spatial_padding_mode)
        self.norm1 = nn.GroupNorm(groups, int(out_channels))
        self.conv2 = nn.Conv1d(int(out_channels), int(out_channels), kernel_size=3, padding=1, padding_mode=spatial_padding_mode)
        self.norm2 = nn.GroupNorm(groups, int(out_channels))
        self.activation = _decoder_activation(activation)
        self.skip = nn.Identity() if int(in_channels) == int(out_channels) else nn.Conv1d(int(in_channels), int(out_channels), kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.skip(value)
        value = self.activation(self.norm1(self.conv1(value)))
        value = self.norm2(self.conv2(value))
        return self.activation(value + residual)

class ProgressiveLowFrequencyPDEDecoder1D(nn.Module):

    def __init__(self, latent_dim: int=192, tokens: int=64, output_points: int=256, output_frames: int=1, *, conv_channels: tuple[int, ...] | list[int] | None=None, blocks_per_stage: int=1, activation: str='gelu', norm_groups: int=8, spatial_padding_mode: str='circular', init_std: float=0.02) -> None:
        super().__init__()
        if tokens < 1 or output_points < tokens or output_points % tokens:
            raise ValueError('decoder output_points must be an integer multiple of tokens')
        ratio = int(output_points) // int(tokens)
        if ratio & ratio - 1:
            raise ValueError('progressive decoder requires a power-of-two upsampling ratio')
        if int(blocks_per_stage) < 1:
            raise ValueError('blocks_per_stage must be positive')
        if int(output_frames) < 1:
            raise ValueError('decoder output_frames must be positive')
        if float(init_std) <= 0.0:
            raise ValueError('init_std must be positive')
        if spatial_padding_mode not in ('circular', 'reflect', 'replicate', 'zeros'):
            raise ValueError('unsupported one-dimensional decoder padding mode')
        self.latent_dim = int(latent_dim)
        self.tokens = int(tokens)
        self.output_points = int(output_points)
        self.output_frames = int(output_frames)
        self.upsampling_stages = int(math.log2(ratio))
        self.blocks_per_stage = int(blocks_per_stage)
        self.spatial_padding_mode = str(spatial_padding_mode)
        self.init_std = float(init_std)
        if conv_channels is None:
            conv_channels = tuple((max(32, self.latent_dim // 2 ** stage) for stage in range(self.upsampling_stages + 1)))
        self.conv_channels = tuple((int(value) for value in conv_channels))
        if any((value < 1 for value in self.conv_channels)):
            raise ValueError('all conv_channels must be positive')
        if len(self.conv_channels) != self.upsampling_stages + 1:
            raise ValueError(f'conv_channels must contain one coarse level plus one level per upsampling stage; expected {self.upsampling_stages + 1}, received {len(self.conv_channels)}')
        self.norm = nn.LayerNorm(self.latent_dim)
        self.input_projection = nn.Linear(self.latent_dim, self.conv_channels[0])
        self.input_blocks = nn.Sequential(*[PeriodicResidualBlock1D(self.conv_channels[0], self.conv_channels[0], activation=activation, norm_groups=norm_groups, spatial_padding_mode=self.spatial_padding_mode) for _ in range(self.blocks_per_stage)])
        self.upsample_blocks = nn.ModuleList()
        for in_channels, out_channels in zip(self.conv_channels[:-1], self.conv_channels[1:]):
            blocks = [PeriodicResidualBlock1D(in_channels, out_channels, activation=activation, norm_groups=norm_groups, spatial_padding_mode=self.spatial_padding_mode)]
            blocks.extend((PeriodicResidualBlock1D(out_channels, out_channels, activation=activation, norm_groups=norm_groups, spatial_padding_mode=self.spatial_padding_mode) for _ in range(self.blocks_per_stage - 1)))
            self.upsample_blocks.append(nn.Sequential(*blocks))
        self.output_head = nn.Conv1d(self.conv_channels[-1], self.output_frames, kernel_size=3, padding=1, padding_mode=self.spatial_padding_mode)
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            if module.weight is not None:
                nn.init.constant_(module.weight, 1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3 or latent.shape[1:] != (self.tokens, self.latent_dim):
            raise ValueError(f'Expected [B,{self.tokens},{self.latent_dim}], received {tuple(latent.shape)}')
        value = self.input_projection(self.norm(latent)).transpose(1, 2)
        value = self.input_blocks(value)
        for blocks in self.upsample_blocks:
            value = F.interpolate(value, scale_factor=2.0, mode='linear', align_corners=False)
            value = blocks(value)
        result = self.output_head(value)
        if result.shape[-1] != self.output_points:
            raise RuntimeError('decoder produced an unexpected spatial resolution')
        return result[:, 0] if self.output_frames == 1 else result
