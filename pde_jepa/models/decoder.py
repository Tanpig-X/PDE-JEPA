# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license in the repository root.
"""Framewise physical-field decoder (paper Appendix B.4)."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import trunc_normal_

def _to_2tuple(value, name):
    if isinstance(value, int):
        result = (value, value)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        result = tuple(value)
    else:
        raise ValueError(f"{name} must be an int or a length-2 tuple, received {value!r}")
    if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in result):
        raise ValueError(f"{name} entries must be positive integers, received {result}")
    return result


def _positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, received {value!r}")
    return value


def _group_count(channels, maximum_groups):
    maximum_groups = min(_positive_int(maximum_groups, "norm_groups"), channels)
    for groups in range(maximum_groups, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _activation(name):
    name = str(name).lower()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"activation must be 'gelu' or 'silu', received {name!r}")


def _initialize_module(module, init_std):
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
        trunc_normal_(module.weight, std=init_std)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        if module.weight is not None:
            nn.init.constant_(module.weight, 1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class SpatialResidualBlock(nn.Module):
    """Spatial residual block used after each progressive 2x upsampling step."""

    def __init__(
        self,
        in_channels,
        out_channels,
        activation="gelu",
        norm_groups=8,
        spatial_padding_mode="circular",
        kernel_size=3,
    ):
        super().__init__()
        in_channels = _positive_int(in_channels, "in_channels")
        out_channels = _positive_int(out_channels, "out_channels")
        if spatial_padding_mode not in ("circular", "reflect", "replicate"):
            raise ValueError(
                "spatial_padding_mode must be one of circular/reflect/replicate, "
                f"received {spatial_padding_mode!r}"
            )
        groups = _group_count(out_channels, norm_groups)
        kernel = _to_2tuple(kernel_size, "kernel_size")
        padding = tuple(size // 2 for size in kernel)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            padding=padding,
            padding_mode=spatial_padding_mode,
        )
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel,
            padding=padding,
            padding_mode=spatial_padding_mode,
        )
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.activation = _activation(activation)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        residual = self.skip(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class FieldDecoder(nn.Module):
    """Decode token-grid features through learned, progressive 2x spatial upsampling."""

    def __init__(
        self,
        img_size=128,
        patch_size=8,
        num_frames=21,
        tubelet_size=1,
        encoder_embed_dim=192,
        out_chans=4,
        conv_channels=None,
        blocks_per_stage=1,
        activation="gelu",
        norm_groups=8,
        spatial_padding_mode="circular",
        init_std=0.02,
        stage_scale_factors=None,
    ):
        super().__init__()
        self.img_height, self.img_width = _to_2tuple(img_size, "img_size")
        patch_height, patch_width = _to_2tuple(patch_size, "patch_size")
        self.patch_size = patch_size
        self.num_frames = _positive_int(num_frames, "num_frames")
        self.tubelet_size = _positive_int(tubelet_size, "tubelet_size")
        self.encoder_embed_dim = _positive_int(encoder_embed_dim, "encoder_embed_dim")
        self.out_chans = _positive_int(out_chans, "out_chans")
        self.blocks_per_stage = _positive_int(blocks_per_stage, "blocks_per_stage")
        self.spatial_padding_mode = str(spatial_padding_mode)
        self.init_std = float(init_std)

        if self.init_std <= 0:
            raise ValueError(f"init_std must be positive, received {init_std}")
        if self.num_frames % self.tubelet_size:
            raise ValueError(f"num_frames={self.num_frames} must be divisible by tubelet_size={self.tubelet_size}")
        if self.img_height % patch_height or self.img_width % patch_width:
            raise ValueError(
                f"img_size={(self.img_height, self.img_width)} must be divisible by patch_size={self.patch_size}"
            )
        if any(size & (size - 1) for size in (patch_height, patch_width)):
            raise ValueError(
                "Progressive low-frequency decoding requires a power-of-two patch_size, "
                f"received {self.patch_size}"
            )

        self.grid_depth = self.num_frames // self.tubelet_size
        self.grid_height = self.img_height // patch_height
        self.grid_width = self.img_width // patch_width
        self.num_patches = self.grid_depth * self.grid_height * self.grid_width
        self.num_upsample_stages = int(math.log2(max(patch_height, patch_width)))
        self.upsample_factors = [tuple(2.0 if 2**stage < size else 1.0
                                       for size in (patch_height, patch_width))
                                 for stage in range(self.num_upsample_stages)]
        if stage_scale_factors is not None:
            self.upsample_factors = [_to_2tuple(value, "stage_scale_factors")
                                     for value in stage_scale_factors]
            if tuple(math.prod(factors[axis] for factors in self.upsample_factors)
                     for axis in (0, 1)) != (patch_height, patch_width):
                raise ValueError("Stage scale factors must multiply to the patch dimensions")
            self.num_upsample_stages = len(self.upsample_factors)
        kernel = (1 if self.img_height == 1 else 3, 1 if self.img_width == 1 else 3)

        if conv_channels is None:
            first_channels = min(self.encoder_embed_dim, 128)
            minimum_channels = max(16, 4 * self.out_chans)
            conv_channels = tuple(
                max(minimum_channels, first_channels // (2**stage))
                for stage in range(self.num_upsample_stages + 1)
            )
        self.conv_channels = tuple(_positive_int(value, "conv_channels") for value in conv_channels)
        expected_channels = self.num_upsample_stages + 1
        if len(self.conv_channels) != expected_channels:
            raise ValueError(
                f"conv_channels must contain {expected_channels} entries for patch_size={self.patch_size}, "
                f"received {self.conv_channels}"
            )

        # Interpolate learned features and apply periodic convolutions after
        # every 2x resize to refine the upsampled field.
        self.norm = nn.LayerNorm(self.encoder_embed_dim)
        self.input_projection = nn.Linear(self.encoder_embed_dim, self.conv_channels[0])
        self.input_blocks = nn.Sequential(
            *[
                SpatialResidualBlock(
                    self.conv_channels[0],
                    self.conv_channels[0],
                    activation=activation,
                    norm_groups=norm_groups,
                    spatial_padding_mode=self.spatial_padding_mode,
                    kernel_size=kernel,
                )
                for _ in range(self.blocks_per_stage)
            ]
        )
        self.upsample_stages = nn.ModuleList()
        for in_channels, out_channels in zip(self.conv_channels[:-1], self.conv_channels[1:]):
            blocks = [
                SpatialResidualBlock(
                    in_channels,
                    out_channels,
                    activation=activation,
                    norm_groups=norm_groups,
                    spatial_padding_mode=self.spatial_padding_mode,
                    kernel_size=kernel,
                )
            ]
            blocks.extend(
                SpatialResidualBlock(
                    out_channels,
                    out_channels,
                    activation=activation,
                    norm_groups=norm_groups,
                    spatial_padding_mode=self.spatial_padding_mode,
                    kernel_size=kernel,
                )
                for _ in range(self.blocks_per_stage - 1)
            )
            self.upsample_stages.append(nn.Sequential(*blocks))

        self.output_head = nn.Conv2d(
            self.conv_channels[-1],
            self.out_chans * self.tubelet_size,
            kernel_size=kernel,
            padding=tuple(size // 2 for size in kernel),
            padding_mode=self.spatial_padding_mode,
        )
        self.apply(lambda module: _initialize_module(module, self.init_std))

    def _validate_tokens(self, x):
        expected = ("B", self.num_patches, self.encoder_embed_dim)
        if x.ndim != 3:
            raise ValueError(f"Expected decoder tokens with shape {expected}, received shape {tuple(x.shape)}")
        if x.shape[1] != self.num_patches or x.shape[2] != self.encoder_embed_dim:
            raise ValueError(
                f"Expected {self.num_patches} tokens with embedding dim {self.encoder_embed_dim}, "
                f"received shape {tuple(x.shape)}"
            )

    def forward(self, x):
        self._validate_tokens(x)
        x = self.input_projection(self.norm(x))
        x = rearrange(
            x,
            "b (t h w) c -> (b t) c h w",
            t=self.grid_depth,
            h=self.grid_height,
            w=self.grid_width,
        )
        x = self.input_blocks(x)
        for stage, factors in zip(self.upsample_stages, self.upsample_factors):
            if factors != (1, 1):
                x = F.interpolate(x, scale_factor=factors, mode="bilinear", align_corners=False)
            x = stage(x)
        x = self.output_head(x)
        return rearrange(
            x,
            "(b t) (c dt) h w -> b c (t dt) h w",
            t=self.grid_depth,
            c=self.out_chans,
            dt=self.tubelet_size,
        )
