# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license; see LICENSE and THIRD_PARTY_NOTICES.md.

import math
import re
from functools import partial
import torch
import torch.nn as nn
import torch.utils.checkpoint
from pde_jepa.masking import apply_masks
from pde_jepa.utils.tensors import trunc_normal_
from pde_jepa.models.encoder_blocks import Block
from pde_jepa.models.patch_embed import PatchEmbed3D, spatial_pair

import os
from collections.abc import Mapping
import yaml
from pde_jepa.utils.checkpoint import robust_checkpoint_loader
from pde_jepa.utils.logging import get_logger

logger = get_logger(__name__)
_WRAPPER_PREFIXES = ("module.", "_orig_mod.", "backbone.")

class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        out_layers=None,
        uniform_power=False,
        use_silu=False,
        wide_silu=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
        is_causal=False,
        use_rope=False,
        init_type: str = "default",
        handle_nonsquare_inputs=True,
        img_temporal_dim_size=None,
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=False,
        modality_embedding=True,
        n_output_distillation=4,
        **kwargs,
    ):
        super().__init__()
        if (not use_rope or use_silu or n_registers or has_cls_first
                or interpolate_rope or img_temporal_dim_size is not None
                or init_type != "default" or out_layers is not None):
            raise ValueError("This encoder uses RoPE without special tokens, interpolation, or alternate output layers")
        if min(in_chans, embed_dim, depth, num_heads, num_frames, tubelet_size) < 1 or embed_dim % num_heads:
            raise ValueError("Encoder dimensions must be positive and embed_dim divisible by num_heads")
        if not 1 <= n_output_distillation <= depth:
            raise ValueError("The number of hierarchical outputs must lie between one and encoder depth")
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers
        self.init_type = init_type
        self.handle_nonsquare_inputs = handle_nonsquare_inputs
        self.img_temporal_dim_size = img_temporal_dim_size

        img_size = spatial_pair(img_size)
        self.patch_shape = spatial_pair(patch_size)
        if any(size % patch for size, patch in zip(img_size, self.patch_shape)):
            raise ValueError("Image dimensions must be multiples of the spatial patch dimensions")
        self.img_height, self.img_width = img_size
        self.grid_height, self.grid_width = (size // patch for size, patch in zip(img_size, self.patch_shape))
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, tubelet_size=tubelet_size,
            in_chans=in_chans, embed_dim=embed_dim,
        )
        self.num_patches = (num_frames // tubelet_size) * self.grid_height * self.grid_width
        self.uniform_power = uniform_power

        self.use_rope = use_rope
        self.blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    grid_depth=num_frames // tubelet_size,
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_sdpa=use_sdpa,
                    is_causal=is_causal,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
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

        self.attn_out = False
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        self.hierarchical_layers = [(index + 1) * depth // n_output_distillation - 1
                                    for index in range(n_output_distillation)]
        self.out_layers_distillation = list(self.hierarchical_layers)

        self.norms_block = nn.ModuleList(
            [norm_layer(embed_dim) for _ in range(len(self.hierarchical_layers))]
        )

        self.return_hierarchical = False
        self.hierarchical_reduction = "concat"

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            return
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, x, masks=None, training=False):
        """
        :param x: input image/video
        :param masks: indices of patch tokens to mask (remove)
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        if x.ndim != 5:
            raise ValueError("Encoder expects [B,C,T,H,W]")
        _, _, T, H, W = x.shape
        T = T // self.tubelet_size
        if H % self.patch_shape[0] or W % self.patch_shape[1] or T < 1:
            raise ValueError("Input must contain complete spatial patches and at least one temporal tube")
        H_patches, W_patches = H // self.patch_shape[0], W // self.patch_shape[1]
        x = self.patch_embed(x)
        mode = "video"

        if masks is not None:
            x = apply_masks(x, masks)
            masks = torch.cat(masks, dim=0)

        hier = []
        for i, blk in enumerate(self.blocks):
            if self.use_activation_checkpointing:
                x, attn = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    masks,
                    T=T,
                    H_patches=H_patches,
                    W_patches=W_patches,
                    use_reentrant=False,
                    return_attn=self.attn_out,
                    mode=mode,
                )
            else:
                x, attn = blk(
                    x,
                    mask=masks,
                    T=T,
                    H_patches=H_patches,
                    W_patches=W_patches,
                    return_attn=self.attn_out,
                    mode=mode,
                )

            if i in self.out_layers_distillation:
                out_idx = self.hierarchical_layers.index(i)
                hier.append(self.norms_block[out_idx](x))

        if training or self.return_hierarchical:
            return torch.cat(hier, dim=2)
        return self.norms_block[-1](x)

def vit_tiny(patch_size=16, **kwargs):
    options = dict(embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
                   qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    options.update(kwargs)
    return VisionTransformer(patch_size=patch_size, **options)


def build_encoder(config, **overrides):
    """Construct the JEPA encoder from data geometry and model dimensions."""
    data, model, meta = config['data'], config['model'], config.get('meta', {})
    dataset = data.get('dataset', data.get('zebra_cfd', {})) or {}
    family = model.get('family', 'vision_transformer')
    if family == 'sequence_1d':
        from pde_jepa.models.sequence_1d import VJEPAEncoder1D
        height, width = spatial_pair(data['crop_size'])
        patch_height, patch_width = spatial_pair(data['patch_size'])
        if height != 1 or patch_height != 1:
            raise ValueError('sequence_1d requires crop_size=[1,X] and patch_size=[1,p]')
        options = dict(
            spatial_points=width, patch_length=patch_width,
            tubelet_size=int(data.get('tubelet_size', 1)),
            in_channels=int(data.get('channels', dataset.get('channels', 1))),
            embed_dim=int(model.get('embed_dim', 192)), depth=int(model.get('depth', 12)),
            num_heads=int(model.get('num_heads', 3)), use_sdpa=bool(meta.get('use_sdpa', True)),
            activation_checkpointing=bool(model.get('use_activation_checkpointing', True)),
        )
        if 'use_activation_checkpointing' in overrides:
            overrides['activation_checkpointing'] = overrides.pop('use_activation_checkpointing')
        options.update(overrides)
        return VJEPAEncoder1D(**options)
    if family != 'vision_transformer':
        raise ValueError(f'Unknown encoder family {family!r}')
    name = model.get('model_name', 'vit_tiny')
    preset = re.fullmatch(r'vit_tiny(?:_depth([1-9][0-9]*))?', name)
    if preset is None:
        raise ValueError(f"Unknown encoder preset {name!r}; configure vit_tiny dimensions explicitly")
    depth = int(preset.group(1) or 12)
    frames = data.get('dataset_fpcs', [data.get('frames', 30)])
    if len(frames) != 1:
        raise ValueError("Encoder configuration requires exactly one frame count")
    options = dict(
        img_size=data['crop_size'], patch_size=data['patch_size'],
        num_frames=int(frames[0]), tubelet_size=int(data.get('tubelet_size', 1)),
        in_chans=int(data.get('channels', dataset.get('channels', 1))),
        embed_dim=int(model.get('embed_dim', 192)), depth=int(model.get('depth', depth)),
        num_heads=int(model.get('num_heads', 3)), mlp_ratio=float(model.get('mlp_ratio', 4)),
        qkv_bias=bool(model.get('qkv_bias', True)),
        uniform_power=bool(model.get('uniform_power', True)),
        use_sdpa=bool(meta.get('use_sdpa', True)), use_rope=bool(model.get('use_rope', True)),
        use_activation_checkpointing=bool(model.get('use_activation_checkpointing', True)),
        use_silu=bool(model.get('use_silu', False)), wide_silu=bool(model.get('wide_silu', True)),
        is_causal=bool(model.get('is_causal', False)), init_type=model.get('init_type', 'default'),
        img_temporal_dim_size=model.get('img_temporal_dim_size'),
        n_registers=int(model.get('n_registers', 0)), has_cls_first=bool(model.get('has_cls_first', False)),
        interpolate_rope=bool(model.get('interpolate_rope', False)),
        modality_embedding=bool(model.get('modality_embedding', False)),
        n_output_distillation=int(model.get('levels_predictor', 4)),
    )
    options.update(overrides)
    return vit_tiny(**options)

def clean_wrapped_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Recursively strip known DDP/compile/backbone wrapper prefixes."""

    cleaned: dict[str, torch.Tensor] = {}
    for original_key, value in state_dict.items():
        key = str(original_key)
        removed = True
        while removed:
            removed = False
            for prefix in _WRAPPER_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    removed = True
                    break
        if key in cleaned:
            raise RuntimeError(
                "Checkpoint prefix cleanup produced duplicate key "
                f"{key!r} from {original_key!r}"
            )
        cleaned[key] = value
    return cleaned

def strict_load_target_encoder(
    encoder: nn.Module,
    checkpoint_path: str,
    *,
    encoder_key: str = "target_encoder",
) -> dict[str, object]:
    """Strictly load the selected encoder weights into the backbone."""

    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Encoder checkpoint does not exist: {checkpoint_path}")
    checkpoint = robust_checkpoint_loader(checkpoint_path, map_location="cpu")
    if encoder_key not in checkpoint:
        raise KeyError(
            f"Checkpoint {checkpoint_path!r} has no {encoder_key!r}; "
            f"available keys={sorted(checkpoint)}"
        )
    state = checkpoint[encoder_key]
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint entry {encoder_key!r} is not a state dictionary")
    cleaned = clean_wrapped_state_dict(state)
    encoder.load_state_dict(cleaned, strict=True)
    info = {
        "checkpoint_path": checkpoint_path,
        "encoder_key": encoder_key,
        "epoch": int(checkpoint.get("epoch", 0)),
        "state_keys": len(cleaned),
    }
    del checkpoint
    return info

def _load_vjepa21_pretrain_config(path: str) -> tuple[dict, str]:
    config_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Encoder pretrain config does not exist: {config_path}")
    with open(config_path) as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    if not isinstance(config, dict):
        raise TypeError(f"Encoder pretrain config must be a mapping: {config_path}")
    if config.get("app", "vjepa_2_1") not in ("vjepa_2_1", "pretrain"):
        raise ValueError(
            "Frozen encoder construction requires a pretraining configuration; "
            f"received app={config.get('app')!r}"
        )
    return config, config_path

def build_frozen_encoder(
    *,
    checkpoint_path: str,
    pretrain_config_path: str,
    encoder_key: str = "target_encoder",
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, object]]:
    """Build the JEPA encoder and strictly load its frozen target weights."""

    pretrain, config_path = _load_vjepa21_pretrain_config(pretrain_config_path)
    data = dict(pretrain.get("data") or {})
    model = dict(pretrain.get("model") or {})
    model_name = str(model.get("model_name", "vit_tiny"))
    frame_counts = data.get("dataset_fpcs", [data.get("frames", 30)])
    if not isinstance(frame_counts, (list, tuple)) or len(frame_counts) != 1:
        raise ValueError("Encoder configuration requires exactly one frame count")
    crop_size = data["crop_size"]
    patch_size = data["patch_size"]
    tubelet_size = int(data.get("tubelet_size", 1))
    num_frames = int(frame_counts[0])
    if tubelet_size != 1:
        raise ValueError("Frame-independent encoding requires pretrained tubelet_size=1")
    encoder = build_encoder(pretrain, use_activation_checkpointing=False)
    in_chans = encoder.patch_embed.proj.in_channels
    encoder.return_hierarchical = False
    encoder.hierarchical_reduction = "concat"
    checkpoint_info = strict_load_target_encoder(
        encoder,
        checkpoint_path,
        encoder_key=encoder_key,
    )
    encoder.requires_grad_(False)
    encoder.eval()
    encoder.to(device)
    if encoder.training or any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("The target encoder was not frozen successfully")

    expected_tokens = encoder.grid_height * encoder.grid_width
    info: dict[str, object] = {
        **checkpoint_info,
        "pretrain_config_path": config_path,
        "pretrain_app": pretrain.get("app", "pretrain"),
        "model_name": model_name,
        "crop_size": crop_size,
        "patch_size": patch_size,
        "tubelet_size": tubelet_size,
        "num_frames": num_frames,
        "in_chans": in_chans,
        "embed_dim": int(encoder.embed_dim),
        "expected_tokens": expected_tokens,
        "grid_shape": [encoder.grid_height, encoder.grid_width],
        "final_latent_only": True,
        "frame_independent": True,
        "forward_training_flag": False,
    }
    logger.info(
        "Loaded frozen target encoder path=%s epoch=%d "
        "model=%s final_shape_per_frame=[%d,%d]",
        info["checkpoint_path"],
        info["epoch"],
        model_name,
        expected_tokens,
        encoder.embed_dim,
    )
    return encoder, info

def encode_frames(
    encoder: nn.Module,
    videos: torch.Tensor,
    *,
    chunk_size: int = 64,
    expected_tokens: int | None = None,
    expected_dim: int | None = None,
) -> torch.Tensor:
    """Encode every frame as an independent one-frame 5-D video clip."""

    if videos.ndim != 5:
        raise ValueError(f"Expected videos [B,C,T,H,W], received {tuple(videos.shape)}")
    if chunk_size < 1:
        raise ValueError("frame encode chunk_size must be positive")
    if encoder.training:
        raise RuntimeError("Frozen encoder must remain in eval mode")
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("Frozen encoder unexpectedly has trainable parameters")
    batch_size, channels, num_frames, height, width = videos.shape
    frames = (
        videos.permute(0, 2, 1, 3, 4)
        .contiguous()
        .reshape(batch_size * num_frames, channels, 1, height, width)
    )
    outputs = []
    with torch.inference_mode():
        for frame_chunk in frames.split(int(chunk_size), dim=0):
            output = encoder(frame_chunk, masks=None, training=False)
            if not torch.is_tensor(output) or output.ndim != 3:
                shape = type(output).__name__ if not torch.is_tensor(output) else tuple(output.shape)
                raise RuntimeError(
                    "The frame encoder must return [BT,N,D], "
                    f"received {shape}"
                )
            outputs.append(output)
        latent = torch.cat(outputs, dim=0)

    # Tensors created in inference_mode cannot be saved by a trainable Linear
    # backward.  Clone outside the context to create a normal, detached tensor.
    latent = latent.clone()
    if expected_tokens is not None and latent.shape[1] != int(expected_tokens):
        raise RuntimeError(
            f"Final latent has {latent.shape[1]} tokens, expected {expected_tokens}; "
            "special-token or patch geometry semantics do not match the pretrain config"
        )
    if expected_dim is not None and latent.shape[2] != int(expected_dim):
        raise RuntimeError(
            f"Final latent dimension is {latent.shape[2]}, expected {expected_dim}; "
            "training=True hierarchical output may have been selected accidentally"
        )
    return latent.reshape(
        batch_size,
        num_frames,
        latent.shape[1],
        latent.shape[2],
    )
