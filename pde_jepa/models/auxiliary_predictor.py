# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license in the repository LICENSE file.

"""Causal auxiliary dynamics model used to train PAG (paper Section 4.2)."""

import math
from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from pde_jepa.models.spatial_blocks import ACBlock, build_action_block_causal_attention_mask
from pde_jepa.utils.tensors import trunc_normal_


class AuxiliaryDynamicsPredictor(nn.Module):
    """Predict latent increments from a causal prefix and PDE conditions.

    This model is discarded after PAG training.
    """

    def __init__(self, *, img_size=128, patch_size=8, num_frames=29,
                 embed_dim=192, predictor_embed_dim=384, depth=24,
                 num_heads=12, use_activation_checkpointing=True,
                 use_sdpa=True, zero_init_output=True, init_std=0.02,
                 condition_dim=1):
        super().__init__()
        spatial = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        patch = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        if len(spatial) != 2 or len(patch) != 2 or any(
                p <= 0 or s <= 0 or s % p for s, p in zip(spatial, patch)):
            raise ValueError("Spatial dimensions must be positive multiples of patch dimensions")
        self.grid_height, self.grid_width = (s // p for s, p in zip(spatial, patch))
        self.condition_dim = int(condition_dim)
        if self.condition_dim < 1:
            raise ValueError("condition_dim must be positive")
        self.num_frames = int(num_frames)
        self.use_activation_checkpointing = bool(use_activation_checkpointing)
        self.init_std = float(init_std)
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.action_encoder = nn.Linear(self.condition_dim, predictor_embed_dim)
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
            self.num_frames, self.grid_height, self.grid_width, add_tokens=1
        ), persistent=False)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x, actions):
        x = self.predictor_embed(x)
        batch, context_tokens, dim = x.shape
        spatial_tokens = self.grid_height * self.grid_width
        frames = context_tokens // spatial_tokens
        if context_tokens % spatial_tokens or not 1 <= frames <= self.num_frames:
            raise ValueError("Input must contain complete frames within the causal context")
        if actions.shape != (batch, frames, self.condition_dim):
            raise ValueError(f"Expected conditions {(batch, frames, self.condition_dim)}, got {tuple(actions.shape)}")
        condition = self.action_encoder(actions).unsqueeze(2)
        x = x.view(batch, frames, spatial_tokens, dim)
        x = torch.cat([condition, x], dim=2).flatten(1, 2)
        attn_mask = self.attn_mask[:x.shape[1], :x.shape[1]].to(x.device, non_blocking=True)
        for block in self.predictor_blocks:
            kwargs = dict(mask=None, attn_mask=attn_mask, T=frames,
                          H=self.grid_height, W=self.grid_width, action_tokens=1)
            if self.training and self.use_activation_checkpointing:
                x = checkpoint(block, x, use_reentrant=False, **kwargs)
            else:
                x = block(x, **kwargs)
        x = x.view(batch, frames, 1 + spatial_tokens, dim)[:, :, 1:].flatten(1, 2)
        return self.predictor_proj(self.predictor_norm(x))
