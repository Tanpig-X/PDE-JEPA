# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license; see LICENSE and THIRD_PARTY_NOTICES.md.

"""Masked JEPA pretraining and cooldown for PDE trajectories."""

import copy
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
import yaml

from pde_jepa.data import build_dataset
from pde_jepa.masking import MaskCollator, apply_masks
from pde_jepa.models.encoder_blocks import Lambda_LinearWarmupHold
from pde_jepa.models.mask_dist import compute_mask_distance
from pde_jepa.training.pretrain_utils import init_opt, init_video_model, load_checkpoint
from pde_jepa.utils.distributed import init_distributed
from pde_jepa.utils.logging import AverageMeter, CSVLogger, get_logger

logger = get_logger(__name__)


def _masked_loss(predictions, targets, masks, exponent, distance_weights=None):
    targets = [apply_masks(target, mask, concat=False) for target, mask in zip(targets, masks)]
    loss, count = 0, 0
    for group_index, (predicted, target) in enumerate(zip(predictions, targets)):
        for mask_index, (prediction, truth) in enumerate(zip(predicted, target)):
            error = torch.abs(prediction - truth) ** exponent
            if distance_weights is not None:
                error = error * (1 / distance_weights[group_index][mask_index].unsqueeze(2))
            loss += error.mean() / exponent
            count += 1
    return loss / count


def train(config: dict):
    """Run JEPA pretraining or cooldown with optional torchrun distribution."""
    data, model = config['data'], config['model']
    meta, optim = config['meta'], config['optimization']
    loss_config = config['loss']
    if (model.get('has_cls_first', False)
            or model.get('normalize_predictor', False)
            or not loss_config.get('predict_all', True)
            or loss_config.get('shift_by_n', 0) != 0):
        raise ValueError('JEPA pretraining requires aligned context/target prediction without CLS tokens')
    folder = Path(config['folder'])
    folder.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda' and int(os.environ.get('WORLD_SIZE', 1)) > 1:
        device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    world_size, rank = init_distributed(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    seed = int(meta.get('seed', 239))
    random.seed(0)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[meta.get('dtype', 'bfloat16')]
    mixed_precision = device.type == 'cuda' and dtype != torch.float32
    if rank == 0:
        with (folder / 'params-pretrain.yaml').open('w') as stream:
            yaml.safe_dump(config, stream, sort_keys=False)

    frame_counts = data.get('dataset_fpcs', [data.get('frames', 30)])
    mask_collator = MaskCollator(
        cfgs_mask=config['mask'], dataset_fpcs=frame_counts,
        crop_size=data['crop_size'], patch_size=data['patch_size'], tubelet_size=data['tubelet_size'],
    )
    encoder, predictor = init_video_model(config, device)
    embed_dim = encoder.backbone.embed_dim
    levels = len(encoder.backbone.hierarchical_layers)
    grid_shape = (encoder.backbone.grid_height, encoder.backbone.grid_width)
    target_encoder = copy.deepcopy(encoder)
    dataset = build_dataset(data, data.get('datasets', data.get('train')),
                            frames_per_clip=frame_counts[0], training=True)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    num_workers = int(data.get('num_workers', 0))
    loader = DataLoader(
        dataset, batch_size=data['batch_size'], sampler=sampler, collate_fn=mask_collator,
        drop_last=True, num_workers=num_workers, pin_memory=data.get('pin_mem', True),
        persistent_workers=num_workers > 0 and data.get('persistent_workers', True),
    )
    ipe = int(optim.get('ipe') or len(loader))
    if ipe < 1:
        raise ValueError('Training dataset must contain at least one full batch per rank')
    epochs = int(optim['epochs'])
    is_anneal = bool(optim.get('is_anneal', False))
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        is_anneal=is_anneal, encoder=encoder, predictor=predictor, iterations_per_epoch=ipe,
        num_epochs=epochs, start_lr=optim['start_lr'], ref_lr=optim['lr'], warmup=optim['warmup'],
        final_lr=optim['final_lr'], wd=optim['weight_decay'], final_wd=optim['final_weight_decay'],
        mixed_precision=mixed_precision, ipe_scale=optim.get('ipe_scale', 1.0),
        betas=tuple(optim.get('betas', (0.9, 0.999))), eps=optim.get('eps', 1e-8),
    )
    if world_size > 1:
        encoder = DistributedDataParallel(encoder, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=False, find_unused_parameters=True)
        target_encoder = DistributedDataParallel(target_encoder)
    target_encoder.requires_grad_(False)
    ema = optim['ema']
    momentum_scheduler = (
        ema[0] + step * (ema[1] - ema[0]) / (ipe * epochs * optim.get('ipe_scale', 1.0))
        for step in range(ipe * epochs + 1)
    )
    lambda_scheduler = Lambda_LinearWarmupHold(model.get('lambda_value_vid', 0.5))
    latest_path = folder / 'latest.pth.tar'
    start_epoch = 0
    if meta.get('load_checkpoint', False):
        resume_anneal = is_anneal and optim.get('resume_anneal', False) and latest_path.exists()
        if is_anneal:
            checkpoint = latest_path if resume_anneal else optim['anneal_ckpt']
        else:
            checkpoint = meta.get('read_checkpoint') or latest_path
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            checkpoint, encoder, predictor, target_encoder, optimizer, scaler,
            is_anneal=is_anneal and not resume_anneal,
        )
        if not is_anneal or resume_anneal:
            for _ in range(start_epoch * ipe):
                scheduler.step()
                wd_scheduler.step()
                next(momentum_scheduler)
                mask_collator.step()

    wandb_run = None
    wandb_config = config.get('wandb', {})
    if rank == 0 and wandb_config.get('enabled', False):
        import wandb
        wandb_run = wandb.init(
            project=wandb_config.get('project', 'PDE-JEPA'), name=wandb_config.get('name'),
            dir=str(folder), config=config, mode=wandb_config.get('mode', 'online'),
        )
    csv_logger = CSVLogger(folder / f'log_r{rank}.csv', ('%d', 'epoch'), ('%d', 'itr'), ('%.8f', 'loss'))
    sampler.set_epoch(start_epoch)
    iterator = iter(loader)
    for epoch in range(start_epoch, epochs):
        meter = AverageMeter()
        started = time.monotonic()
        for iteration in range(ipe):
            try:
                sample = next(iterator)
            except StopIteration:
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                sample = next(iterator)
            clips = [batch[0][0][0].to(device, non_blocking=True) for batch in sample]
            masks_enc = [[mask.to(device, non_blocking=True) for mask in batch[1]] for batch in sample]
            masks_pred = [[mask.to(device, non_blocking=True) for mask in batch[2]] for batch in sample]
            lr, wd = scheduler.step(), wd_scheduler.step()
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=mixed_precision):
                with torch.no_grad():
                    targets = target_encoder(clips, training_mode=True)
                    targets = [torch.cat([
                        F.layer_norm(value[:, :, level * embed_dim:(level + 1) * embed_dim], (embed_dim,))
                        for level in range(levels)
                    ], dim=2) for value in targets]
                context = encoder(clips, masks_enc, training_mode=True)
                pred_target, pred_context = predictor(context, masks_enc, masks_pred)
                loss = _masked_loss(pred_target, targets, masks_pred, loss_config['loss_exp'])
                distances = compute_mask_distance(
                    masks_pred, masks_enc, grid_shape,
                    loss_config.get('offset_context_loss', False),
                ) if loss_config.get('weight_distance_loss', True) else None
                context_loss = _masked_loss(pred_context, targets, masks_enc, loss_config['loss_exp'], distances)
                weight = lambda_scheduler.value(epoch * ipe + iteration) if model.get('lambda_progressive', True) else model['lambda_value_vid']
                loss = loss + context_loss * weight
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite pretraining loss at epoch={epoch + 1}, iteration={iteration}')
            if mixed_precision:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            momentum = min(next(momentum_scheduler), ema[1])
            with torch.no_grad():
                target_params = list(target_encoder.parameters())
                torch._foreach_mul_(target_params, momentum)
                torch._foreach_add_(target_params, list(encoder.parameters()), alpha=1 - momentum)
            value = float(loss.detach())
            meter.update(value)
            csv_logger.log(epoch + 1, iteration, value)
            if rank == 0 and (iteration % 10 == 0 or iteration == ipe - 1):
                logger.info('epoch=%d iteration=%d/%d loss=%.6f lr=%.3e', epoch + 1, iteration, ipe, meter.avg, lr)
                if wandb_run is not None:
                    wandb_run.log({'train/loss': value, 'optimizer/lr': lr, 'optimizer/wd': wd, 'epoch': epoch + 1}, step=epoch * ipe + iteration)
        if rank == 0:
            checkpoint = {
                'encoder': encoder.state_dict(), 'predictor': predictor.state_dict(),
                'target_encoder': target_encoder.state_dict(), 'opt': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler is not None else None,
                'epoch': epoch + 1, 'loss': meter.avg, 'batch_size': data['batch_size'],
                'world_size': world_size, 'lr': optim['lr'],
            }
            torch.save(checkpoint, latest_path)
            save_every = int(meta.get('save_every_freq', 10))
            if save_every > 0 and (epoch + 1) % save_every == 0:
                torch.save(checkpoint, folder / f'e{epoch}.pth.tar')
            logger.info('Epoch %d finished in %.1fs, loss=%.6f', epoch + 1, time.monotonic() - started, meter.avg)
    if wandb_run is not None:
        wandb_run.finish()
