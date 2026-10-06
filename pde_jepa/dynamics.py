"""Task-specific conditions, state initialization, and latent transitions."""

from contextlib import nullcontext
import math

import torch

from pde_jepa.integrator import RK4Diagnostics, fixed_step_rk4
from pde_jepa.models.psp import build_psp


CONDITION_DIMS = {
    'vorticity': 1, 'advection': 1, 'wave2d': 2, 'grayscott': 2,
    'wave_b': 4, 'burgers': 7, 'heat': 7, 'combined': 3,
}


def task_name(config, model_config=None):
    model = config.get('model', {}) if model_config is None else model_config
    name = str(model.get('task', config.get('task', 'vorticity'))).lower().replace('-', '_')
    name = {'wave_2d': 'wave2d', 'gray_scott': 'grayscott'}.get(name, name)
    if name not in CONDITION_DIMS:
        raise ValueError(f'Unknown PDE task {name!r}')
    return name


def build_predictor(model_config, device, *, task=None, activation_checkpointing=None):
    model = dict(model_config)
    for key in ('type', 'expected_trainable_parameters', 'parameter_structure',
                'state_dim', 'vector_field', 'damping_branch', 'spatial_periodicity'):
        model.pop(key, None)
    selected = task or task_name({'model': model})
    model.pop('task', None)
    if selected == 'grayscott' and 'num_tokens' in model:
        if model.pop('num_tokens') != math.prod(model.get('grid_size', (8, 8))):
            raise ValueError('Gray-Scott num_tokens must match grid_size')
    if activation_checkpointing is not None:
        key = 'use_activation_checkpointing' if selected in ('burgers', 'heat') else 'activation_checkpointing'
        model[key] = bool(activation_checkpointing)
    return build_psp(selected, **model).to(device)


def prepare_conditions(task, conditions, statistics=None):
    """Convert raw task parameters or prepared forcing features to model inputs."""
    values = conditions.float()
    if values.ndim == 1:
        values = values.unsqueeze(-1)
    if values.ndim not in (2, 3) or values.shape[-1] != CONDITION_DIMS[task]:
        raise ValueError(f'{task} conditions must be [B,{CONDITION_DIMS[task]}] or [B,T-1,{CONDITION_DIMS[task]}]')
    if not bool(torch.isfinite(values).all()):
        raise ValueError('Conditions contain NaN or Inf')
    statistics = statistics or {}
    if task == 'vorticity':
        from pde_jepa.models.psp import RawNuStats
        return RawNuStats(statistics['nu_min_train'], statistics['nu_max_train']).transform(values)
    transform = statistics.get('transform', 'identity')
    if transform == 'standardize':
        mean = torch.as_tensor(statistics['mean'], device=values.device, dtype=values.dtype)
        std = torch.as_tensor(statistics['std'], device=values.device, dtype=values.dtype)
        if mean.shape != (values.shape[-1],) or std.shape != mean.shape:
            raise ValueError('Condition mean/std must match the parameter dimension')
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
            raise ValueError('Condition statistics must be finite with positive standard deviations')
        values = (values - mean) / std
    elif transform != 'identity':
        raise ValueError(f'Unknown condition transform {transform!r}')
    return values


def transition_condition(conditions, frame):
    return conditions[:, frame] if conditions.ndim == 3 else conditions


def phase_state(previous, current, velocity_scale):
    if previous.shape != current.shape or not math.isfinite(float(velocity_scale)) or velocity_scale <= 0:
        raise ValueError('Phase states require matching latent shapes and a positive velocity scale')
    return torch.cat((current.float(), (current.float() - previous.float()) / float(velocity_scale)), dim=-1)


def validate_transition(task, delta_t, solver):
    direct = task in ('burgers', 'heat')
    expected = 'direct' if direct else 'rk4'
    if solver != expected:
        raise ValueError(f'{task} requires the {expected} transition rule')
    if not math.isfinite(float(delta_t)) or delta_t <= 0:
        raise ValueError('The data interval must be finite and positive')
    if task == 'wave_b' and float(delta_t) != 1.0:
        raise ValueError('Wave-B phase dynamics use a unit normalized data interval')


def step_predictor(model, state, conditions, *, delta_t=1.0, substeps=4,
                   autocast_dtype=torch.bfloat16, return_diagnostics=False,
                   derivative_dtype="float32", state_dtype="float32", step_size_mode="scalar"):
    def field(value, parameters):
        context = torch.autocast(value.device.type, dtype=autocast_dtype) if (
            value.device.type == 'cuda' and autocast_dtype != torch.float32) else nullcontext()
        with context:
            return model(value.float() if value.device.type != 'cuda' or autocast_dtype == torch.float32 else value,
                         parameters)
    if state_dtype not in ('float32', 'native'):
        raise ValueError('state_dtype must be float32 or native')
    if state_dtype == 'float32':
        state = state.float()
    if getattr(model, 'is_continuous', True):
        return fixed_step_rk4(field, state, conditions.float(), delta_t=delta_t,
                              substeps=substeps, return_diagnostics=return_diagnostics,
                              derivative_dtype=derivative_dtype, step_size_mode=step_size_mode)
    prediction = field(state, conditions.float())
    if state_dtype == 'float32':
        prediction = prediction.float()
    if not return_diagnostics:
        return prediction
    rms = lambda value: value.square().mean().sqrt()
    diagnostics = RK4Diagnostics(rms(prediction - state), torch.maximum(rms(state), rms(prediction)),
                                 torch.count_nonzero(~torch.isfinite(prediction)))
    return prediction, diagnostics
