# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license; see LICENSE and THIRD_PARTY_NOTICES.md.

import logging
import pde_jepa.models.jepa_predictor as vit_pred
from pde_jepa.models.encoder import build_encoder
import torch
from pde_jepa.pretrain_wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper
from pde_jepa.utils.checkpoint import robust_checkpoint_loader
from pde_jepa.utils.schedulers import (
    CosineWDSchedule,
    LinearDecaySchedule,
    WarmupCosineSchedule,
)

logger = logging.getLogger(__name__)

def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
    is_anneal=False,
):
    logger.info(f"Loading {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = 0
    if not is_anneal:
        epoch = checkpoint["epoch"]

    def restore(module, key):
        from pde_jepa.models.encoder import clean_wrapped_state_dict
        source = clean_wrapped_state_dict(checkpoint[key])
        target_keys = module.state_dict()
        state = {}
        for name in target_keys:
            clean = next(iter(clean_wrapped_state_dict({name: target_keys[name]})))
            state[name] = source[clean]
        module.load_state_dict(state, strict=True)

    restore(encoder, "encoder")
    restore(predictor, "predictor")
    if target_encoder is not None:
        restore(target_encoder, "target_encoder")

    try:
        opt.load_state_dict(checkpoint["opt"])
    except ValueError:
        print("[warn] Optimizer groups mismatch; reinitializing optimizer.")
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    logger.info(f"loaded optimizers from epoch {epoch}")
    logger.info(f"read-path: {r_path}")
    del checkpoint

    return (
        encoder,
        predictor,
        target_encoder,
        opt,
        scaler,
        epoch,
    )

def init_video_model(config, device):
    """Build trainable JEPA modules with shared, configuration-defined geometry."""
    data, model, meta = config['data'], config['model'], config.get('meta', {})
    encoder = MultiSeqWrapper(build_encoder(config))
    frame_counts = data.get('dataset_fpcs', [data.get('frames', 30)])
    predictor = vit_pred.vit_predictor(
        img_size=data['crop_size'], patch_size=data['patch_size'],
        num_frames=max(frame_counts), tubelet_size=data.get('tubelet_size', 1),
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=model.get('pred_embed_dim', 384),
        depth=model.get('pred_depth', 12), num_heads=model.get('pred_num_heads', 12),
        n_output_distillation=len(encoder.backbone.hierarchical_layers),
        uniform_power=model.get('uniform_power', True),
        use_mask_tokens=True, num_mask_tokens=len(config['mask']) * len(frame_counts),
        zero_init_mask_tokens=model.get('zero_init_mask_tokens', True),
        use_rope=model.get('use_rope', True), use_sdpa=meta.get('use_sdpa', True),
        is_causal=model.get('pred_is_causal', False),
        use_activation_checkpointing=model.get('use_activation_checkpointing', True),
        return_all_tokens=True, modality_embedding=False,
    )
    predictor = PredictorMultiSeqWrapper(predictor)
    encoder.to(device)
    predictor.to(device)
    logger.info("Encoder parameters: %d; predictor parameters: %d",
                sum(p.numel() for p in encoder.parameters()),
                sum(p.numel() for p in predictor.parameters()))
    return encoder, predictor

def init_opt(
    is_anneal,
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
):
    param_groups = [
        {
            "params": (
                p
                for n, p in encoder.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in encoder.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)

    if not is_anneal:
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    else:
        scheduler = LinearDecaySchedule(
            optimizer,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )

    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler

