"""Train Physics-Structured Latent Predictor with endpoint supervision only."""

import csv
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from pde_jepa.dynamics import (build_predictor, phase_state, prepare_conditions, step_predictor,
                               task_name, transition_condition, validate_transition)
from pde_jepa.integrator import RK4Diagnostics
from pde_jepa.rollout import resolve_dtype
from pde_jepa.training.latent_cache import ProjectedQCache, cache_split_dir, prepare_cache
from pde_jepa.utils.checkpoint import robust_checkpoint_loader


class UniformTransitionDataset(Dataset):
    """Uniformly sample transitions without replacement from each trajectory."""

    def __init__(self, cache_root, *, transitions_per_trajectory=4, seed=0,
                 task="vorticity", velocity_scale=None):
        self.cache_root = Path(cache_root)
        cache = ProjectedQCache(self.cache_root)
        self.task, self.velocity_scale = task, velocity_scale
        self.trajectory_count, self.num_frames = cache.shape[:2]
        self.offset = int(task == "wave_b")
        self.num_frames -= self.offset
        self.transitions_per_trajectory, self.seed = int(transitions_per_trajectory), int(seed)
        if not 1 <= self.transitions_per_trajectory < self.num_frames:
            raise ValueError('Invalid number of transitions per trajectory')
        self._cache = None
        self.set_epoch(0)

    def set_epoch(self, epoch):
        rng = np.random.default_rng(self.seed + int(epoch))
        trajectory = np.repeat(np.arange(self.trajectory_count, dtype=np.int64), self.transitions_per_trajectory)
        time = np.empty(len(trajectory), dtype=np.int64)
        for index in range(self.trajectory_count):
            start = index * self.transitions_per_trajectory
            time[start:start + self.transitions_per_trajectory] = rng.choice(
                self.num_frames - 1, size=self.transitions_per_trajectory, replace=False)
        permutation = rng.permutation(len(trajectory))
        self._trajectory, self._time = trajectory[permutation], time[permutation]

    def __len__(self):
        return len(self._trajectory)

    def __getitem__(self, index):
        if self._cache is None:
            self._cache = ProjectedQCache(self.cache_root)
        trajectory, frame = int(self._trajectory[index]), int(self._time[index]) + self.offset
        source, target = self._cache.frame(trajectory, frame), self._cache.frame(trajectory, frame + 1)
        if self.task == "wave_b":
            previous = self._cache.frame(trajectory, frame - 1)
            source, target = (phase_state(previous, source, self.velocity_scale),
                              phase_state(source, target, self.velocity_scale))
        return source, target, self._cache.condition(trajectory, frame), frame, trajectory


def build_generator(config, device, *, activation_checkpointing=False):
    return build_predictor(config['model'], device, task=task_name(config),
                           activation_checkpointing=activation_checkpointing)


def integrate_generator(generator, q, conditions, *, delta_t, substeps,
                        autocast_dtype=torch.bfloat16, derivative_dtype="float32",
                        state_dtype="float32", step_size_mode="scalar"):
    if (isinstance(delta_t, torch.Tensor) and delta_t.numel() > 1 and
            derivative_dtype == 'native' and step_size_mode == 'scalar' and
            getattr(generator, 'is_continuous', True)):
        values, rows, checks = [], [], []
        for interval in delta_t.unique():
            indices = torch.nonzero(delta_t == interval, as_tuple=True)[0]
            value, check = integrate_generator(
                generator, q[indices], conditions[indices], delta_t=float(interval),
                substeps=substeps, autocast_dtype=autocast_dtype, derivative_dtype=derivative_dtype,
                state_dtype=state_dtype, step_size_mode=step_size_mode)
            values.append(value)
            rows.append(indices)
            checks.append(check)
        diagnostics = RK4Diagnostics(
            torch.stack([c.max_substep_increment_rms for c in checks]).max(),
            torch.stack([c.max_intermediate_state_rms for c in checks]).max(),
            torch.stack([c.nonfinite_count for c in checks]).sum())
        return torch.cat(values)[torch.cat(rows).argsort()], diagnostics
    return step_predictor(generator, q, conditions, delta_t=delta_t, substeps=substeps,
                          autocast_dtype=autocast_dtype, return_diagnostics=True,
                          derivative_dtype=derivative_dtype, state_dtype=state_dtype,
                          step_size_mode=step_size_mode)


def _transition_interval(config, frames, device=None):
    intervals = config['data'].get('time_deltas')
    if intervals is None:
        return float(config['data']['delta_t_data'])
    if isinstance(frames, int):
        return float(intervals[frames])
    values = torch.as_tensor(intervals, dtype=torch.float32, device=device)
    return values[frames.to(device=device, dtype=torch.long)]


def _predictor_dtype(config):
    return resolve_dtype(config.get('precision', {}).get(
        'predictor', config['representation'].get('dtype', 'bfloat16')))


def train_step_chunks(*, config, generator, optimizer, chunks, device, autocast_dtype):
    """The normalization denominator is pooled over the entire effective batch."""
    optimizer.zero_grad(set_to_none=True)
    task = task_name(config)
    target_delta_sum2, total_numel = 0.0, 0
    for source, target, *_ in chunks:
        delta = target.float() - source.float()
        target_delta_sum2 += float(delta.square().sum())
        total_numel += delta.numel()
    denominator = target_delta_sum2 / total_numel + float(config['loss']['eps'])
    loss_sum = 0.0
    for source, target, parameters, frames, *_ in chunks:
        source, target = source.to(device), target.to(device).float()
        if config['ode'].get('state_dtype', 'float32') == 'float32':
            source = source.float()
        conditions = prepare_conditions(task, parameters.to(device), config.get("condition"))
        prediction, diagnostics = integrate_generator(
            generator, source, conditions, delta_t=_transition_interval(config, frames, device),
            substeps=config['ode']['substeps'], autocast_dtype=autocast_dtype,
            derivative_dtype=config['ode'].get('derivative_dtype', 'float32'),
            state_dtype=config['ode'].get('state_dtype', 'float32'),
            step_size_mode=config['ode'].get('step_size_mode', 'scalar'))
        loss = (prediction - target).square().sum() / (total_numel * denominator)
        source_rms = source.float().square().mean().sqrt().clamp_min(1e-12)
        if (not bool(torch.isfinite(loss)) or int(diagnostics.nonfinite_count) or
            diagnostics.max_intermediate_state_rms > 10 * source_rms or
            diagnostics.max_substep_increment_rms > 5 * source_rms):
            raise FloatingPointError('Nonfinite or unstable PSP integration')
        loss.backward()
        loss_sum += float(loss.detach())
    grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), config['training']['grad_clip'],
                                              error_if_nonfinite=True)
    optimizer.step()
    return {'train_loss': loss_sum, 'grad_norm': float(grad_norm)}


@torch.inference_mode()
def evaluate_teacher_one_step(config, generator, device, indices=None):
    """ID validation: average per-trajectory, per-transition latent NRMSE."""
    generator.eval()
    cache = ProjectedQCache(cache_split_dir(config, 'id_val'))
    indices = np.arange(len(cache)) if indices is None else np.asarray(indices)
    batch_size = int(config['validation']['batch_size'])
    task = task_name(config)
    dtype = _predictor_dtype(config)
    sums = np.zeros(cache.shape[1] - 1 - int(task == 'wave_b'), dtype=np.float64)
    for start in range(0, len(indices), batch_size):
        subset = indices[start:start + batch_size]
        q = torch.stack([cache.trajectory(int(i)) for i in subset]).to(device)
        conditions = prepare_conditions(task, torch.tensor(np.array(cache.conditions[subset]),
                                        dtype=torch.float32, device=device), config.get('condition'))
        offset = int(task == 'wave_b')
        states = phase_state(q[:, :-1], q[:, 1:], config['model']['velocity_scale']) if offset else q
        for index in range(states.shape[1] - 1):
            frame = index + offset
            target = states[:, index + 1].float()
            prediction, _ = integrate_generator(generator, states[:, index],
                                                    transition_condition(conditions, frame),
                                                    delta_t=_transition_interval(config, frame),
                                                    substeps=config['ode']['substeps'], autocast_dtype=dtype,
                                                    derivative_dtype=config['ode'].get('derivative_dtype', 'float32'),
                                                    state_dtype=config['ode'].get('state_dtype', 'float32'),
                                                    step_size_mode=config['ode'].get('step_size_mode', 'scalar'))
            numerator = (prediction - target).square().flatten(1).sum(1)
            denominator = target.square().flatten(1).sum(1).clamp_min(1e-12)
            sums[index] += float(torch.sqrt(numerator / denominator).sum())
    return float((sums / len(indices)).mean())


def _scheduler(config, optimizer, total_steps):
    settings = config['training']
    minimum = settings['min_lr'] / settings['lr']
    warmup = max(1, int(round(settings['warmup_ratio'] * total_steps)))
    def multiplier(step):
        if step < warmup:
            return minimum + (1.0 - minimum) * step / warmup
        progress = min(max((step - warmup) / max(total_steps - warmup, 1), 0.0), 1.0)
        return minimum + 0.5 * (1.0 - minimum) * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def train(config: dict):
    task = task_name(config)
    config['model']['task'] = task
    ode = config.setdefault('ode', {})
    ode.setdefault('solver', 'direct' if task in ('burgers', 'heat') else 'rk4')
    ode.setdefault('state_dtype', 'float32')
    ode.setdefault('step_size_mode', 'scalar')
    ode.setdefault('substeps', 4)
    config['data'].setdefault('delta_t_data', 1.0)
    validate_transition(task, config['data']['delta_t_data'], ode['solver'])
    if ode['state_dtype'] not in ('float32', 'native'):
        raise ValueError('state_dtype must be float32 or native')
    if ode['step_size_mode'] not in ('scalar', 'batch_tensor'):
        raise ValueError('step_size_mode must be scalar or batch_tensor')
    intervals = config['data'].get('time_deltas')
    if intervals is not None:
        if len(intervals) != int(config['data']['frames']) - 1:
            raise ValueError('time_deltas must contain one value per frame interval')
        for delta in intervals:
            validate_transition(task, delta, ode['solver'])
    spec = config['representation']
    precision = config.setdefault('precision', {})
    precision.setdefault('representation', spec.get('dtype', 'bfloat16'))
    precision.setdefault('predictor', spec.get('dtype', 'bfloat16'))
    storage_dtype = spec.get('storage_dtype', precision.get('initial_latent', 'bfloat16'))
    if 'initial_latent' in precision and precision['initial_latent'] != storage_dtype:
        raise ValueError('initial_latent precision must match representation.storage_dtype')
    precision['initial_latent'] = storage_dtype
    spec.update(dtype=precision['representation'], storage_dtype=storage_dtype)
    for name in ('representation', 'predictor', 'initial_latent'):
        resolve_dtype(precision[name])
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('PSP uses a single process with gradient accumulation')
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    for split in ('id_train', 'id_val'):
        prepare_cache(config, split, device)
    cache = ProjectedQCache(cache_split_dir(config, 'id_train'))
    config['representation']['pag_sha256'] = cache.metadata['pag_sha256']
    config['representation']['encoder_sha256'] = cache.metadata['encoder_sha256']
    validation_cache = ProjectedQCache(cache_split_dir(config, 'id_val'))
    for key in ('pag_sha256', 'encoder_sha256'):
        if validation_cache.metadata[key] != cache.metadata[key]:
            raise ValueError('ID train and validation caches use different representation weights')
    if task == 'vorticity':
        from pde_jepa.models.psp import RawNuStats
        stats = RawNuStats(float(cache.conditions.min()), float(cache.conditions.max()))
        config['condition'] = stats.as_dict()
    else:
        config.setdefault('condition', {'transform': 'identity'})
    if task == 'wave_b' and 'velocity_scale' not in config['model']:
        sum2, count = 0.0, 0
        for index in range(len(cache)):
            q = cache.trajectory(index).float()
            delta = q[1:] - q[:-1]
            sum2 += float(delta.double().square().sum())
            count += delta.numel()
        config['model']['velocity_scale'] = math.sqrt(sum2 / count)
    settings = config['training']
    seed = int(settings['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    dataset = UniformTransitionDataset(cache_split_dir(config, 'id_train'), seed=seed,
                                       transitions_per_trajectory=settings['transitions_per_trajectory'],
                                       task=task, velocity_scale=config['model'].get('velocity_scale'))
    microbatch, effective = int(settings['microbatch']), int(settings['effective_transition_batch_size'])
    if effective % microbatch or len(dataset) % effective:
        raise ValueError('PSP dataset and microbatch must form complete effective batches')
    accumulation = effective // microbatch
    loader = DataLoader(dataset, batch_size=microbatch, shuffle=False, drop_last=False,
                        num_workers=config['data']['num_workers'], pin_memory=device.type == 'cuda',
                        persistent_workers=False)
    epochs, steps_per_epoch = int(settings['epochs']), len(dataset) // effective
    generator = build_generator(config, device, activation_checkpointing=settings.get('activation_checkpointing', False))
    optimizer = torch.optim.AdamW(generator.parameters(), lr=settings['lr'], betas=tuple(settings['betas']),
                                 weight_decay=settings['weight_decay'])
    scheduler = _scheduler(config, optimizer, epochs * steps_per_epoch)
    output = Path(config['folder'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'resolved_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    latest, best_path = output / 'checkpoints/latest.pth.tar', output / 'checkpoints/best_id_val.pth.tar'
    start_epoch, resume_batch, global_step, best, best_epoch = 0, 0, 0, math.inf, None
    resume = settings.get('resume_checkpoint') or (str(latest) if settings.get('resume', True) and latest.exists() else None)
    if resume:
        state = robust_checkpoint_loader(resume, map_location=device)
        if task_name({}, state['model_config']) != task:
            raise ValueError('Cannot resume a predictor trained for a different PDE task')
        if state.get('condition_config', config['condition']) != config['condition']:
            raise ValueError('Cannot resume with different condition statistics')
        protocol = {'state_dtype': ode['state_dtype'], 'step_size_mode': ode['step_size_mode'],
                    'derivative_dtype': ode.get('derivative_dtype', 'float32'),
                    'precision': precision, 'time_deltas': intervals}
        for name, value in protocol.items():
            if name in state and state[name] != value:
                raise ValueError(f'Cannot resume with a different {name}')
        generator.load_state_dict(state['generator'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch, resume_batch = int(state['resume_epoch_index']), int(state['batch_in_epoch'])
        global_step, best = int(state['global_step']), float(state['best_id_val_metric'])
        best_epoch = state.get('best_id_val_epoch')
    def payload(epoch, epoch_index, batch):
        return {'generator': generator.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'epoch': epoch, 'resume_epoch_index': epoch_index,
                'batch_in_epoch': batch, 'global_step': global_step, 'best_id_val_metric': best,
                'best_id_val_epoch': best_epoch,
                **(config['condition'] if task == 'vorticity' else {}),
                'condition_config': config['condition'], 'delta_t_data': config['data']['delta_t_data'],
                'ode_solver': ode['solver'], 'ode_substeps': ode['substeps'],
                'derivative_dtype': ode.get('derivative_dtype', 'float32'),
                'state_dtype': ode['state_dtype'], 'step_size_mode': ode['step_size_mode'],
                'precision': precision, 'time_deltas': intervals,
                'forward_semantics': 'next_state' if ode['solver'] == 'direct' else 'structured_ode_rk4',
                'model_config': config['model'],
                'latent_representation': config['representation'], 'run_kind': 'formal'}
    val_count = len(ProjectedQCache(cache_split_dir(config, 'id_val')))
    subset = np.rint(np.linspace(0, val_count - 1, min(val_count, config['validation']['subset_every_epoch']))).astype(np.int64)
    dtype = _predictor_dtype(config)
    run = None
    if config.get('wandb', {}).get('enabled', False):
        import wandb
        wb = config['wandb']
        run = wandb.init(project=wb['project'], name=wb.get('name', f'{task}-psp'), dir=str(output),
                         config=config, mode=wb.get('mode', 'online'))
    for epoch in range(start_epoch, epochs):
        dataset.set_epoch(epoch)
        generator.train()
        chunks, losses = [], []
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < resume_batch:
                continue
            chunks.append(batch)
            if len(chunks) < accumulation:
                continue
            metrics = train_step_chunks(config=config, generator=generator, optimizer=optimizer,
                                        chunks=chunks, device=device, autocast_dtype=dtype)
            chunks = []
            scheduler.step()
            global_step += 1
            losses.append(metrics['train_loss'])
            if global_step % settings['log_every_steps'] == 0:
                print(f'PSP epoch={epoch+1} step={global_step} loss={metrics["train_loss"]:.6f}', flush=True)
                if run:
                    run.log(metrics, step=global_step)
            if global_step % settings['checkpoint_every_steps'] == 0:
                _save(latest, payload(epoch, epoch, batch_index + 1))
        subset_metric = evaluate_teacher_one_step(config, generator, device, subset)
        full_metric = None
        if (epoch + 1) % config['validation']['full_every_epochs'] == 0:
            full_metric = evaluate_teacher_one_step(config, generator, device)
            if full_metric < best:
                best, best_epoch = full_metric, epoch + 1
                _save(best_path, payload(epoch+1, epoch+1, 0))
        _save(latest, payload(epoch+1, epoch+1, 0))
        row = {'epoch': epoch+1, 'global_step': global_step,
               'mean_train_loss': float(np.mean(losses)) if losses else None,
               'id_val_subset_mean_state_nrmse': subset_metric,
               'id_val_full_mean_state_nrmse': full_metric, 'best_id_val_metric': best}
        history = output / 'training_history.csv'
        exists = history.exists()
        with history.open('a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(f'PSP epoch={epoch+1} ID-val subset={subset_metric:.7f} full={full_metric}', flush=True)
        if run:
            run.log({k: v for k, v in row.items() if v is not None}, step=global_step)
    (output / 'TRAINING_COMPLETE').write_text('complete\n')
    if run:
        run.finish()
