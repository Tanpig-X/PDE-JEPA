"""Frozen JEPA → PAG → PSP rollout used to train and validate the field decoder."""

from contextlib import nullcontext
import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from pde_jepa.dynamics import (build_predictor, phase_state, prepare_conditions, step_predictor,
                               task_name, transition_condition, validate_transition)
from pde_jepa.models.encoder import build_frozen_encoder, encode_frames
from pde_jepa.models.pag import PhysicsAlignedGeometry


def state_sha256(state):
    """Hash parameter names, shapes, dtypes, and values to identify model weights."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def resolve_dtype(name):
    dtypes = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
    if name not in dtypes:
        raise ValueError(f'Unsupported numerical precision {name!r}')
    return dtypes[name]


def autocast(device, dtype=torch.bfloat16):
    """Apply the selected CUDA precision, with FP32 operations on CPU."""
    return torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32) if device.type == "cuda" else nullcontext()


class FrozenRollout:
    def __init__(self, config, device):
        self.device = torch.device(device)
        rep = config["representation"]
        self.encoder, self.encoder_info = build_frozen_encoder(
            checkpoint_path=rep["encoder_checkpoint"],
            pretrain_config_path=rep["encoder_pretrain_config"],
            encoder_key=rep.get("encoder_key", "target_encoder"), device=self.device,
        )
        self.num_tokens = int(self.encoder_info["expected_tokens"])
        self.latent_dim = int(self.encoder_info["embed_dim"])
        pag_ck = torch.load(rep["pag_checkpoint"], map_location="cpu", weights_only=False)
        hidden_dim = rep.get("projector_hidden_dim", pag_ck["projector"]["fc1.weight"].shape[0])
        self.pag = PhysicsAlignedGeometry(dim=self.latent_dim, hidden_dim=hidden_dim).to(self.device)
        self.pag.load_state_dict(pag_ck["projector"], strict=True)
        psp_ck = torch.load(config["psp_checkpoint"], map_location="cpu", weights_only=False)
        model = dict(psp_ck.get("model_config", psp_ck.get("config", {}).get("model", config.get('predictor', {}))))
        if not model:
            raise ValueError("Predictor checkpoint must contain its model configuration")
        requested = config.get("task", config.get("model", {}).get("task"))
        if requested is not None:
            if "task" in model and task_name({"task": requested}) != task_name({}, model):
                raise ValueError("Configured PDE task differs from the predictor checkpoint")
            model["task"] = requested
        self.task = task_name(config, model)
        if self.task == 'wave_b' and 'velocity_scale' not in model:
            model['velocity_scale'] = psp_ck['velocity_scale']
        self.psp = build_predictor(model, self.device, task=self.task)
        key = config.get("psp_state_key", "generator" if "generator" in psp_ck else "predictor")
        self.psp.load_state_dict(psp_ck[key], strict=True)
        self.condition_config = dict(psp_ck.get("condition_config", {}))
        if not self.condition_config:
            self.condition_config = ({key: psp_ck[key] for key in ("nu_min_train", "nu_max_train")}
                                     if self.task == "vorticity" else dict(config.get("condition", {})))
        direct = not getattr(self.psp, "is_continuous", True)
        self.delta_t = float(psp_ck.get("delta_t_data", psp_ck.get("data_interval", config["data"].get("delta_t_data", 1.0))))
        self.substeps = int(psp_ck.get("ode_substeps", config.get("ode", {}).get("substeps", 4)))
        solver = psp_ck.get("ode_solver", "direct" if direct else "rk4")
        validate_transition(self.task, self.delta_t, solver)
        self.derivative_dtype = psp_ck.get("derivative_dtype", config.get("ode", {}).get("derivative_dtype", "float32"))
        self.state_dtype = psp_ck.get('state_dtype', config.get('ode', {}).get('state_dtype', 'float32'))
        self.step_size_mode = psp_ck.get('step_size_mode', config.get('ode', {}).get('step_size_mode', 'scalar'))
        source_rep = psp_ck.get('latent_representation', {})
        precision = {**psp_ck.get('precision', {}), **config.get('precision', {})}
        self.representation_dtype = resolve_dtype(precision.get('representation', source_rep.get('dtype', 'bfloat16')))
        self.predictor_dtype = resolve_dtype(precision.get('predictor', 'bfloat16'))
        self.initial_dtype = resolve_dtype(precision.get('initial_latent', source_rep.get('storage_dtype', 'bfloat16')))
        self.storage_dtype = resolve_dtype(precision.get('rollout_latent', 'bfloat16'))
        self.observed_frames = 2 if self.task == "wave_b" else 1
        self.frames = int(config["data"]["frames"])
        self.time_deltas = (psp_ck.get('time_deltas') or config['data'].get('time_deltas')
                            or [self.delta_t] * (self.frames - 1))
        if len(self.time_deltas) != self.frames - 1:
            raise ValueError('time_deltas must contain one value per frame interval')
        for delta in self.time_deltas:
            validate_transition(self.task, delta, solver)
        if self.frames <= self.observed_frames:
            raise ValueError("A rollout needs at least one frame after the initial observations")
        pag_digest = state_sha256(self.pag.state_dict())
        expected_pag = source_rep.get("geometry_projector_sha256", source_rep.get("pag_sha256"))
        if expected_pag is not None and pag_digest != expected_pag:
            raise ValueError("PAG weights differ from the PSP training representation")
        encoder_digest = state_sha256(self.encoder.state_dict())
        expected_encoder = source_rep.get("encoder_sha256")
        if expected_encoder is not None and encoder_digest != expected_encoder:
            raise ValueError("Encoder weights differ from the PSP training representation")
        self.provenance = {
            "encoder_sha256": encoder_digest,
            "pag_sha256": pag_digest,
            "psp_sha256": state_sha256(psp_ck[key]),
            "encoder_epoch": self.encoder_info.get("epoch"),
            "pag_epoch": pag_ck.get("epoch"), "psp_epoch": psp_ck.get("epoch"),
            "task": self.task, "observed_frames": self.observed_frames,
            "condition_config": self.condition_config,
            "delta_t": self.delta_t, "substeps": self.substeps, "solver": solver,
            "derivative_dtype": self.derivative_dtype,
            "network_dtype": str(self.predictor_dtype).split('.')[-1] if self.device.type == "cuda" else "float32",
            "state_dtype": self.state_dtype, "stored_latent_dtype": str(self.storage_dtype).split('.')[-1],
            "precision": precision, "time_deltas": self.time_deltas,
            "step_size_mode": self.step_size_mode,
        }
        for module in (self.encoder, self.pag, self.psp):
            module.eval().requires_grad_(False)
        self.frame_encode_chunk_size = int(rep.get("frame_encode_chunk_size", 64))

    @torch.inference_mode()
    def __call__(self, videos, parameters):
        """Evolve from the task's initial observations at the configured precision."""
        if videos.shape[2] < self.observed_frames:
            raise ValueError("Not enough initial frames for the selected PDE task")
        with autocast(self.device, self.representation_dtype):
            z0 = encode_frames(self.encoder, videos[:, :, :self.observed_frames],
                               chunk_size=self.frame_encode_chunk_size,
                               expected_tokens=self.num_tokens, expected_dim=self.latent_dim)
            initial = self.pag(z0).to(self.initial_dtype)
        if self.state_dtype == 'float32':
            initial = initial.float()
        conditions = prepare_conditions(self.task, parameters.to(self.device), self.condition_config)
        if conditions.ndim == 3 and conditions.shape[1] != self.frames - 1:
            raise ValueError("Time-dependent conditions must contain frames-1 intervals")
        states = list(initial.to(self.storage_dtype).unbind(1))
        current = (phase_state(initial[:, 0], initial[:, 1], float(self.psp.velocity_scale))
                   if self.task == "wave_b" else initial[:, 0])
        for frame in range(self.observed_frames - 1, self.frames - 1):
            current = step_predictor(self.psp, current, transition_condition(conditions, frame),
                                     delta_t=self.time_deltas[frame], substeps=self.substeps,
                                     derivative_dtype=self.derivative_dtype, state_dtype=self.state_dtype,
                                     autocast_dtype=self.predictor_dtype, step_size_mode=self.step_size_mode)
            if not torch.isfinite(current).all():
                raise FloatingPointError("PSP rollout contains NaN or Inf")
            q = current[..., :self.latent_dim] if self.task == "wave_b" else current
            states.append(q.to(self.storage_dtype))
        return torch.stack(states, dim=1)


def decoder_tokens(q, device, normalize=True):
    """Optionally normalize latent features and flatten their frame-token axes."""
    with autocast(torch.device(device)):
        tokens = (F.layer_norm(q, (q.shape[-1],)) if normalize else q).flatten(1, 2)
    return tokens if torch.device(device).type == "cuda" else tokens.float()
