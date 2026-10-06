"""Train Physics-Aligned Latent Geometry with its auxiliary dynamics predictor."""

import csv
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from pde_jepa.data import build_dataset
from pde_jepa.models.auxiliary_predictor import AuxiliaryDynamicsPredictor
from pde_jepa.models.encoder import build_frozen_encoder, encode_frames
from pde_jepa.models.pag import PhysicsAlignedGeometry
from pde_jepa.training.pag_losses import normalized_mse, temporal_geometry_matching_loss
from pde_jepa.training.pag_utils import normalize_condition, predictor_teacher_deltas
from pde_jepa.utils.checkpoint import robust_checkpoint_loader
from pde_jepa.utils.distributed import init_distributed
from pde_jepa.utils.schedulers import CosineWDSchedule, WarmupCosineSchedule


def _unwrap(module):
    return module.module if isinstance(module, DDP) else module


def _batch(sample, device):
    videos = sample[0][0].to(device, non_blocking=True)
    parameters = sample[1].to(device, dtype=torch.float32, non_blocking=True)
    return videos, parameters


def _loader(config, training, rank, world_size):
    data = config['data']
    source = data.get('datasets' if training else 'val_datasets',
                      data.get('train' if training else 'val'))
    dataset = build_dataset(data, source, training=training)
    if training:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        dataset = Subset(dataset, range(rank, len(dataset), world_size))
        sampler = None
    workers = int(data['num_workers'])
    loader = DataLoader(dataset, batch_size=data['batch_size'], sampler=sampler,
                        shuffle=False, drop_last=training, num_workers=workers,
                        pin_memory=bool(data.get('pin_mem', True)),
                        persistent_workers=workers > 0 and bool(data.get('persistent_workers', True)))
    return loader, sampler


def compute_losses(pixels, parameters, *, encoder, projector, predictor, config, encoder_info):
    """Compute the PAG objective with detached target increments and encoder features."""
    z = encode_frames(encoder, pixels,
                      chunk_size=int(config['encoder']['frame_encode_chunk_size']),
                      expected_tokens=int(encoder_info['expected_tokens']),
                      expected_dim=int(encoder_info['embed_dim']))
    q = projector(z)
    geometry, _ = temporal_geometry_matching_loss(pixels, q, **config['geometry'])
    condition = normalize_condition(parameters, config.get('parameter_normalization', config.get('condition', {})))
    prediction = predictor_teacher_deltas(predictor, q, condition)
    target = (q[:, 1:] - q[:, :-1]).detach()
    prediction_loss = normalized_mse(prediction, target, eps=config['loss']['eps'])
    anchor = normalized_mse(q, z.detach(), eps=config['loss']['eps'])
    loss = prediction_loss + config['loss']['lambda_geometry'] * geometry + config['loss']['lambda_anchor'] * anchor
    return loss, {'loss': loss.detach(), 'prediction_loss': prediction_loss.detach(),
                  'geometry_loss': geometry.detach(), 'anchor_loss': anchor.detach()}


def _optimizer(projector, predictor, config, total_steps):
    decay, no_decay = [], []
    for model in (projector, predictor):
        for name, parameter in model.named_parameters():
            (no_decay if 'bias' in name or parameter.ndim == 1 else decay).append(parameter)
    opt = config['optimization']
    optimizer = torch.optim.AdamW([
        {'params': decay}, {'params': no_decay, 'WD_exclude': True, 'weight_decay': 0.0},
    ], lr=opt['start_lr'], betas=tuple(opt['betas']), eps=opt['eps'])
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=int(opt['warmup'] * total_steps / opt['epochs']),
                                    start_lr=opt['start_lr'], ref_lr=opt['lr'],
                                    final_lr=opt['final_lr'], T_max=total_steps)
    wd = CosineWDSchedule(optimizer, ref_wd=opt['weight_decay'],
                         final_wd=opt['final_weight_decay'], T_max=total_steps)
    return optimizer, scheduler, wd


@torch.no_grad()
def _validate(loader, *, encoder, projector, predictor, config, device, dtype, encoder_info):
    projector, predictor = _unwrap(projector), _unwrap(predictor)
    projector.eval()
    predictor.eval()
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    limit = math.ceil(config['evaluation']['max_trajectories'] / (config['data']['batch_size'] * (dist.get_world_size() if dist.is_initialized() else 1)))
    for batch_index, sample in enumerate(loader):
        if batch_index >= limit:
            break
        pixels, parameters = _batch(sample, device)
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == 'cuda' and dtype != torch.float32):
            _, metrics = compute_losses(pixels, parameters, encoder=encoder, projector=projector,
                                        predictor=predictor, config=config, encoder_info=encoder_info)
        totals[:4] += torch.stack(list(metrics.values())).double() * pixels.shape[0]
        totals[4] += pixels.shape[0]
    if dist.is_initialized():
        dist.all_reduce(totals)
    if totals[4] == 0:
        raise ValueError('ID validation has no trajectories')
    return dict(zip(('loss', 'prediction_loss', 'geometry_loss', 'anchor_loss'),
                    (totals[:4] / totals[4]).cpu().tolist()))


def train(config: dict):
    world_size, rank = init_distributed(config.get("device"))
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda':
        if world_size > 1:
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(device)
    seed = int(config['meta']['seed']) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    dtype = getattr(torch, config['meta']['dtype'])
    folder = Path(config['folder'])
    folder.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (folder / 'params-pretrain.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    loader, sampler = _loader(config, True, rank, world_size)
    val_loader, _ = _loader(config, False, rank, world_size)
    encoder, info = build_frozen_encoder(
        checkpoint_path=config['meta']['encoder_checkpoint'],
        pretrain_config_path=config['meta']['encoder_pretrain_config'],
        device=device,
    )
    projector = PhysicsAlignedGeometry(dim=info['embed_dim'], **config['projector']).to(device)
    predictor_options = dict(config['predictor'])
    condition_dim = loader.dataset.condition_dim
    if int(predictor_options.get('condition_dim', condition_dim)) != condition_dim:
        raise ValueError('Auxiliary predictor condition_dim must match the training parameters')
    predictor_options['condition_dim'] = condition_dim
    predictor = AuxiliaryDynamicsPredictor(img_size=info['crop_size'], patch_size=info['patch_size'],
                                          num_frames=loader.dataset.frames - 1,
                                          embed_dim=info['embed_dim'], **predictor_options).to(device)
    effective = int(config['optimization'].get('effective_batch_size', 16))
    local_batch = int(config['data']['batch_size'])
    if effective % (world_size * local_batch):
        raise ValueError('PAG effective batch must be divisible by world_size * batch_size')
    accumulation = effective // (world_size * local_batch)
    steps_per_epoch = len(loader) // accumulation
    if steps_per_epoch < 1 or len(loader) % accumulation:
        raise ValueError('PAG training data must form complete effective batches')
    epochs = int(config['optimization']['epochs'])
    optimizer, scheduler, wd = _optimizer(projector, predictor, config, epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and dtype == torch.float16)
    epoch_start, global_step, best = 0, 0, math.inf
    resume = config['meta'].get('read_checkpoint')
    if resume is None and config['meta'].get('load_checkpoint') and (folder / 'latest.pt').exists():
        resume = folder / 'latest.pt'
    if resume:
        state = robust_checkpoint_loader(resume, map_location='cpu')
        projector.load_state_dict(state['projector'], strict=True)
        predictor.load_state_dict(state['predictor'], strict=True)
        optimizer.load_state_dict(state['opt'])
        scaler.load_state_dict(state['scaler'])
        epoch_start, global_step = int(state['epoch']), int(state['global_step'])
        best = float(state['best_val_loss'])
        for _ in range(global_step):
            scheduler.step()
            wd.step()
    if world_size > 1:
        projector = DDP(projector, broadcast_buffers=False, find_unused_parameters=False)
        predictor = DDP(predictor, broadcast_buffers=False, find_unused_parameters=False)
    run = None
    if rank == 0 and config.get('wandb', {}).get('enabled', False):
        import wandb
        settings = config['wandb']
        run = wandb.init(project=settings['project'], name=settings.get('name', 'pag'),
                         dir=str(folder), config=config, mode=settings.get('mode', 'online'))
    for epoch in range(epoch_start, epochs):
        sampler.set_epoch(epoch)
        projector.train()
        predictor.train()
        iterator = iter(loader)
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            lr = scheduler.step()
            wd.step()
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for micro in range(accumulation):
                pixels, parameters = _batch(next(iterator), device)
                sync = micro == accumulation - 1
                pag_context = projector.no_sync() if isinstance(projector, DDP) and not sync else nullcontext()
                aux_context = predictor.no_sync() if isinstance(predictor, DDP) and not sync else nullcontext()
                with pag_context, aux_context:
                    with torch.autocast(device.type, dtype=dtype, enabled=device.type == 'cuda' and dtype != torch.float32):
                        loss, _ = compute_losses(pixels, parameters, encoder=encoder, projector=projector,
                                                 predictor=predictor, config=config, encoder_info=info)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Nonfinite PAG loss')
                    scaler.scale(loss / accumulation).backward()
                step_loss += float(loss.detach()) / accumulation
            scaler.unscale_(optimizer)
            parameters = [p for group in optimizer.param_groups for p in group['params']]
            torch.nn.utils.clip_grad_norm_(parameters, config['optimization']['clip_grad'], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            epoch_loss += step_loss
            if rank == 0 and global_step % config['meta']['log_freq'] == 0:
                print(f'PAG epoch={epoch+1} step={global_step} loss={step_loss:.6f} lr={lr:.3e}', flush=True)
                if run:
                    run.log({'train/loss': step_loss, 'lr': lr}, step=global_step)
        validation = _validate(val_loader, encoder=encoder, projector=projector, predictor=predictor,
                               config=config, device=device, dtype=dtype, encoder_info=info)
        if rank == 0:
            improved = validation['loss'] < best
            best = min(best, validation['loss'])
            payload = {'projector': _unwrap(projector).state_dict(), 'predictor': _unwrap(predictor).state_dict(),
                       'opt': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch+1,
                       'global_step': global_step, 'best_val_loss': best, 'args': config, 'encoder_info': info}
            for name in ['latest.pt'] + (['best.pt'] if improved else []):
                temporary = folder / (name + '.tmp')
                torch.save(payload, temporary)
                temporary.replace(folder / name)
            row = {'epoch': epoch+1, 'train_loss': epoch_loss / steps_per_epoch, **validation}
            history = folder / 'training_history.csv'
            exists = history.exists()
            with history.open('a', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
            print(f'PAG epoch={epoch+1} ID-val loss={validation["loss"]:.6f}', flush=True)
            if run:
                run.log({f'val/{k}': v for k, v in validation.items()}, step=global_step)
        if dist.is_initialized():
            dist.barrier()
    if run:
        run.finish()
