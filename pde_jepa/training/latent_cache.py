"""Resumable PAG trajectories and task conditions for predictor training."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pde_jepa.data import build_dataset
from pde_jepa.dynamics import CONDITION_DIMS, task_name
from pde_jepa.models.encoder import build_frozen_encoder, encode_frames
from pde_jepa.models.pag import PhysicsAlignedGeometry
from pde_jepa.rollout import resolve_dtype, state_sha256
from pde_jepa.utils.checkpoint import robust_checkpoint_loader


def cache_split_dir(config, split):
    return Path(config.get('latent_cache_root', Path(config['folder']) / 'latent_cache')) / split


def _storage_spec(name):
    formats = {'bfloat16': ('projected_q_bf16_bits.npy', np.uint16),
               'float16': ('projected_q_f16.npy', np.float16),
               'float32': ('projected_q_f32.npy', np.float32)}
    if name not in formats:
        raise ValueError('Latent storage_dtype must be bfloat16, float16, or float32')
    return formats[name]


def _signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def _write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def build_representation(config, device):
    spec = config['representation']
    encoder, info = build_frozen_encoder(checkpoint_path=spec['encoder_checkpoint'],
                                         pretrain_config_path=spec['encoder_pretrain_config'],
                                         encoder_key=spec.get('encoder_key', 'target_encoder'),
                                         device=device)
    state = robust_checkpoint_loader(spec['pag_checkpoint'], map_location='cpu')
    hidden_dim = spec.get('projector_hidden_dim', state['projector']['fc1.weight'].shape[0])
    projector = PhysicsAlignedGeometry(dim=info['embed_dim'], hidden_dim=hidden_dim).to(device)
    projector.load_state_dict(state['projector'], strict=True)
    projector.eval().requires_grad_(False)
    return encoder, projector, info


@torch.inference_mode()
def prepare_cache(config, split, device=None, representation=None):
    """Cache ID trajectories with constant or interval-dependent conditions."""
    if split not in ('id_train', 'id_val'):
        raise ValueError('PSP training caches accept only ID train and ID validation')
    device = torch.device(device or config.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu'))
    spec, data = config['representation'], config['data']
    task = task_name(config)
    source = data['train' if split == 'id_train' else 'id_val']
    dataset = build_dataset(data, source, training=False, random_temporal_crop=False)
    root = cache_split_dir(config, split)
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / 'cache_metadata.json'
    previous = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if representation is None and (not data.get('tokens') or not data.get('latent_dim')):
        representation = build_representation(config, device)
    tokens = data.get('tokens', representation[2]['expected_tokens'] if representation else None)
    latent_dim = data.get('latent_dim', representation[2]['embed_dim'] if representation else None)
    q_shape = [len(dataset), int(data['frames']), int(tokens), int(latent_dim)]
    sample_condition = dataset[0][1]
    if sample_condition.ndim not in (1, 2) or sample_condition.shape[-1] != CONDITION_DIMS[task]:
        raise ValueError(f'{task} dataset conditions must end in {CONDITION_DIMS[task]} parameters')
    if sample_condition.ndim == 2 and sample_condition.shape[0] != q_shape[1] - 1:
        raise ValueError('Interval conditions must contain frames-1 rows')
    condition_shape = [len(dataset), *sample_condition.shape]
    precision = config.get('precision', {})
    dtype = resolve_dtype(precision.get('representation', spec.get('dtype', 'bfloat16')))
    if device.type != 'cuda':
        dtype = torch.float32
    storage_name = spec.get('storage_dtype', precision.get('initial_latent', 'bfloat16'))
    q_filename, storage_numpy = _storage_spec(storage_name)
    storage_torch = resolve_dtype(storage_name)
    source_signature = (_signature(source) if isinstance(source, (str, Path)) and Path(source).is_file()
                        else [_signature(path) for path in dataset.files])
    expected = {'q_shape': q_shape, 'split': split, 'source': source_signature,
                'encoder': _signature(spec['encoder_checkpoint']),
                'encoder_config': _signature(spec['encoder_pretrain_config']),
                'projector': _signature(spec['pag_checkpoint']),
                'channel_mean': data.get('channel_mean'), 'channel_std': data.get('channel_std'),
                'network_dtype': str(dtype), 'storage_dtype': storage_name,
                'task': task, 'condition_shape': condition_shape, 'data_config': data}
    expected = json.loads(json.dumps(expected))
    if previous:
        optional = {'task', 'condition_shape', 'data_config'} if task == 'vorticity' else set()
        if any(previous.get(key) != value for key, value in expected.items()
               if key not in optional or key in previous):
            raise ValueError(f'Cache provenance differs at {root}; choose a new cache directory')
        if (root / 'EXTRACTION_COMPLETE').exists():
            return previous
    else:
        _write_json(metadata_path, expected)
    q_path, complete_path = root / q_filename, root / 'completed_bool.npy'
    conditions_path = root / 'conditions_f32.npy'
    if q_path.exists() != complete_path.exists():
        raise RuntimeError(f'Incomplete cache allocation at {root}; choose a new directory')
    if q_path.exists():
        q_map = np.load(q_path, mmap_mode='r+')
        completed = np.load(complete_path, mmap_mode='r+')
        if list(q_map.shape) != q_shape or q_map.dtype != storage_numpy or completed.shape != (len(dataset),):
            raise ValueError(f'Invalid cache array shape/dtype at {root}')
    else:
        q_map = np.lib.format.open_memmap(q_path, mode='w+', dtype=storage_numpy, shape=tuple(q_shape))
        completed = np.lib.format.open_memmap(complete_path, mode='w+', dtype=np.bool_, shape=(len(dataset),))
        completed[:] = False
        completed.flush()
    if conditions_path.exists():
        conditions = np.load(conditions_path, mmap_mode='r+')
        if list(conditions.shape) != condition_shape or conditions.dtype != np.float32:
            raise ValueError('Invalid cached condition shape or dtype')
    else:
        conditions = np.lib.format.open_memmap(conditions_path, mode='w+', dtype=np.float32,
                                             shape=tuple(condition_shape))
        if bool(completed.any()):
            if task != 'vorticity' or not (root / 'nu_f32.npy').exists():
                raise ValueError('Completed cache rows have no saved conditions')
            conditions[:] = np.load(root / 'nu_f32.npy').reshape(condition_shape)
        conditions.flush()
    if representation is None:
        representation = build_representation(config, device)
    encoder, projector, info = representation
    if (info['expected_tokens'], info['embed_dim']) != (tokens, latent_dim):
        raise ValueError('Configured latent shape differs from the encoder output')
    expected['pag_sha256'] = state_sha256(projector.state_dict())
    expected['encoder_sha256'] = state_sha256(encoder.state_dict())
    missing = np.flatnonzero(~completed)
    subset = torch.utils.data.Subset(dataset, missing.tolist())
    loader = DataLoader(subset, batch_size=data.get('cache_batch_size', 8), shuffle=False,
                        num_workers=data.get('num_workers', 4), pin_memory=device.type == 'cuda')
    offset = 0
    for batch_index, sample in enumerate(loader):
        videos = sample[0][0].to(device, non_blocking=True)
        values = sample[1].float()
        if not bool(torch.isfinite(values).all()):
            raise ValueError('Nonfinite task parameters in cache extraction')
        if task == 'vorticity' and not bool((values > 0).all()):
            raise ValueError('Viscosities must be positive')
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == 'cuda' and dtype != torch.float32):
            z = encode_frames(encoder, videos, chunk_size=spec.get('frame_encode_chunk_size', 64),
                              expected_tokens=tokens, expected_dim=latent_dim)
            q = projector(z)
        if not bool(torch.isfinite(q).all()):
            raise FloatingPointError('Nonfinite projected latent in cache extraction')
        indices = missing[offset:offset + len(videos)]
        stored = q.to(storage_torch).contiguous().cpu()
        if storage_torch == torch.bfloat16:
            stored = stored.view(torch.uint16)
        q_map[indices] = stored.numpy()
        conditions[indices] = values.numpy()
        q_map.flush()
        conditions.flush()
        completed[indices] = True
        completed.flush()
        offset += len(videos)
        if batch_index % 20 == 0:
            print(f'PAG cache {split}: {int(completed.sum())}/{len(dataset)} trajectories', flush=True)
    _write_json(metadata_path, {**expected, 'complete': True})
    (root / 'EXTRACTION_COMPLETE').write_text('complete\n')
    return {**expected, 'complete': True}


class ProjectedQCache:
    def __init__(self, root):
        self.root = Path(root)
        if not (self.root / 'EXTRACTION_COMPLETE').is_file():
            raise FileNotFoundError(f'Incomplete PAG cache: {root}')
        self.metadata = json.loads((self.root / 'cache_metadata.json').read_text())
        self.shape = tuple(self.metadata['q_shape'])
        self.storage_dtype = self.metadata.get('storage_dtype', 'bfloat16')
        self.q_filename, self.numpy_dtype = _storage_spec(self.storage_dtype)
        path = self.root / 'conditions_f32.npy'
        self.conditions = (np.load(path, mmap_mode='r') if path.exists()
                           else np.load(self.root / 'nu_f32.npy', mmap_mode='r').reshape(-1, 1))
        if self.conditions.shape[0] != self.shape[0] or self.conditions.ndim not in (2, 3):
            raise ValueError('Cached conditions do not match the trajectories')
        self._q = None

    def __len__(self):
        return self.shape[0]

    @property
    def q(self):
        if self._q is None:
            self._q = np.load(self.root / self.q_filename, mmap_mode='r')
            if self._q.shape != self.shape or self._q.dtype != self.numpy_dtype:
                raise ValueError('Cached latent shape or dtype differs from its metadata')
        return self._q

    def condition(self, trajectory, frame):
        values = self.conditions[trajectory]
        if values.ndim == 2:
            values = values[frame]
        return torch.from_numpy(np.array(values, copy=True)).float()

    def frame(self, trajectory, frame):
        value = torch.from_numpy(np.array(self.q[trajectory, frame], copy=True))
        return value.view(torch.bfloat16) if self.storage_dtype == 'bfloat16' else value

    def trajectory(self, index):
        value = torch.from_numpy(np.array(self.q[index], copy=True))
        return value.view(torch.bfloat16) if self.storage_dtype == 'bfloat16' else value
