# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint

from pde_jepa.masking import apply_masks
from pde_jepa.utils.tensors import trunc_normal_

from pde_jepa.models.encoder_blocks import Block
from pde_jepa.models.patch_embed import spatial_pair


class VisionTransformerPredictor(nn.Module):
    """Vision Transformer Predictor"""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=384,
        out_embed_dim=None,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=False,
        use_mask_tokens=False,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_silu=False,
        wide_silu=True,
        is_causal=False,
        use_activation_checkpointing=False,
        return_all_tokens=False,
        chop_last_n_tokens=0,
        use_rope=False,
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=False,
        modality_embedding=True,
        img_temporal_dim_size=None,
        teacher_embed_dim=None,
        **kwargs
    ):
        super().__init__()
        self.return_all_tokens = return_all_tokens
        self.chop_last_n_tokens = chop_last_n_tokens
        self.has_cls_first = has_cls_first

        if not use_rope or use_silu or not use_mask_tokens or not return_all_tokens:
            raise ValueError("JEPA prediction requires RoPE, GELU, mask tokens and context prediction")
        if min(depth, num_heads, embed_dim, predictor_embed_dim, num_mask_tokens) < 1 or predictor_embed_dim % num_heads:
            raise ValueError("Predictor dimensions must be positive and width divisible by the head count")
        n_output_distillation = int(kwargs.get("n_output_distillation", 4))
        if n_output_distillation < 1:
            raise ValueError("The predictor requires at least one hierarchical output")
        self.hierarchical_layers = [(index + 1) * depth // n_output_distillation - 1
                                    for index in range(n_output_distillation)]

        act_layer_mlp = nn.SiLU if use_silu else nn.GELU
        if len(self.hierarchical_layers) == 1:
            self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        else:
            self.predictor_embed = nn.Sequential(
                nn.Linear(embed_dim * len(self.hierarchical_layers), embed_dim, bias=True),
                act_layer_mlp(),
                nn.Linear(embed_dim, predictor_embed_dim, bias=True),
            )

        self.mask_tokens = None
        self.num_mask_tokens = 0
        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            self.mask_tokens = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
                    for i in range(num_mask_tokens)
                ]
            )

        img_size = spatial_pair(img_size)
        self.patch_shape = spatial_pair(patch_size)
        if any(size % patch for size, patch in zip(img_size, self.patch_shape)):
            raise ValueError("Image dimensions must be multiples of the spatial patch dimensions")
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.grid_height = img_size[0] // self.patch_shape[0]
        self.grid_width = img_size[1] // self.patch_shape[1]
        self.grid_depth = num_frames // self.tubelet_size
        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.num_patches = self.grid_depth * self.grid_height * self.grid_width

        self.uniform_power = uniform_power

        self.use_rope = use_rope
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    grid_depth=self.grid_depth,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_sdpa=bool(kwargs.get("use_sdpa", True)),
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    is_causal=is_causal,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    n_registers=n_registers,
                    has_cls_first=has_cls_first,
                    interpolate_rope=interpolate_rope,
                    patch_size=patch_size,
                )
                for i in range(depth)
            ]
        )

        if out_embed_dim is None:
            if teacher_embed_dim is not None:
                out_embed_dim = teacher_embed_dim // len(self.hierarchical_layers)
            else:
                out_embed_dim = embed_dim
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(
            predictor_embed_dim,
            len(self.hierarchical_layers) * out_embed_dim,
            bias=True,
        )
        if self.return_all_tokens:
            self.predictor_proj_context = nn.Linear(
                predictor_embed_dim,
                out_embed_dim * len(self.hierarchical_layers),
                bias=True,
            )

        self.init_std = init_std
        if not zero_init_mask_tokens:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)

        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, x, masks_x, masks_y, mask_index=1):
        """
        :param x: context tokens
        :param masks_x: indices of context tokens in input
        :params masks_y: indices of target tokens in input
        """
        assert (masks_x is not None) and (
            masks_y is not None
        ), "Cannot run predictor without mask indices"
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks_y, list):
            masks_y = [masks_y]

        B = len(x) // len(masks_x)

        x = self.predictor_embed(x)
        _, N_ctxt, D = x.shape

        mask_index = mask_index % self.num_mask_tokens
        pred_tokens = self.mask_tokens[mask_index]
        pred_tokens = pred_tokens.repeat(B, self.num_patches, 1)
        pred_tokens = apply_masks(pred_tokens, masks_y)

        x = x.repeat(len(masks_x), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        masks_x = torch.cat(masks_x, dim=0)
        masks_y = torch.cat(masks_y, dim=0)
        masks = torch.cat([masks_x, masks_y], dim=1)

        argsort = torch.argsort(masks, dim=1)
        masks = torch.stack([masks[i, row] for i, row in enumerate(argsort)], dim=0)
        x = torch.stack([x[i, row, :] for i, row in enumerate(argsort)], dim=0)

        if self.chop_last_n_tokens > 0:
            x = x[:, : -self.chop_last_n_tokens]
            masks = masks[:, : -self.chop_last_n_tokens]

        for i, blk in enumerate(self.predictor_blocks):
            geometry = dict(T=self.grid_depth, H_patches=self.grid_height, W_patches=self.grid_width)
            if self.use_activation_checkpointing:
                x, attn = torch.utils.checkpoint.checkpoint(
                    blk, x, masks, use_reentrant=False, **geometry
                )
            else:
                x, attn = blk(x, mask=masks, **geometry)
        x = self.predictor_norm(x)

        reverse_argsort = torch.argsort(argsort, dim=1)
        x = torch.stack(
            [x[i, row, :] for i, row in enumerate(reverse_argsort)], dim=0
        )
        x_pred = x[:, N_ctxt:, :]
        x_context = x[:, :N_ctxt, :]
        x_pred = self.predictor_proj(x_pred)
        x_context = self.predictor_proj_context(x_context)
        return x_pred, x_context


def vit_predictor(**kwargs):
    model = VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
