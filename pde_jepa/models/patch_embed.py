# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license; see LICENSE and THIRD_PARTY_NOTICES.md.


from numbers import Integral

import torch.nn as nn


def spatial_pair(value):
    """Return positive (height, width) dimensions from an integer or pair."""
    if isinstance(value, Integral):
        pair = (value, value)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        pair = tuple(value)
    else:
        raise ValueError("Spatial dimensions must be an integer or a two-element list/tuple")
    if any(not isinstance(size, Integral) or isinstance(size, bool) or size < 1 for size in pair):
        raise ValueError("Spatial dimensions must be positive integers")
    return tuple(int(size) for size in pair)


class PatchEmbed3D(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(
        self,
        patch_size=16,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        self.patch_size = spatial_pair(patch_size)
        if not isinstance(tubelet_size, Integral) or isinstance(tubelet_size, bool) or tubelet_size < 1:
            raise ValueError("tubelet_size must be a positive integer")
        self.tubelet_size = tubelet_size

        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, *self.patch_size),
            stride=(tubelet_size, *self.patch_size),
        )

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

