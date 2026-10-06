"""PDE trajectories and physical parameters stored in HDF5."""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PDETrajectoryDataset(Dataset):
    """Read [N,C,H,W,T] or [N,C,X,T] as normalized [C,T,H,W].

    One-dimensional fields use H=1. Parameters are [P], or [T-1,P] when
    already prepared for the selected transition times. Multiple HDF5 keys
    are concatenated on the last parameter axis without normalization.
    """

    def __init__(
        self, root_path, group_name="auto", states_key="states", condition_key="mu",
        frames_per_clip=30, temporal_stride=1, spatial_stride=1, trajectory_stride=1,
        record_indices=None, channels=1, channel_mean=None, channel_std=None,
        training=True, random_temporal_crop=False, file_pattern="*.h5",
        condition_dim=None,
    ):
        if root_path is None:
            raise ValueError("A PDE trajectory source path is required")
        paths = [root_path] if isinstance(root_path, (str, Path)) else root_path
        self.files = sorted({str(f) for p in paths for f in
                             ([Path(p)] if Path(p).is_file() else Path(p).glob(file_pattern))})
        if not self.files:
            raise FileNotFoundError(f"No HDF5 trajectories at {root_path}")
        self.states_key, self.condition_key = states_key, condition_key
        self.condition_keys = ([condition_key] if isinstance(condition_key, str)
                               else list(condition_key or []))
        if any(not isinstance(key, str) or not key for key in self.condition_keys):
            raise ValueError("condition_key must be a string or a list of dataset names")
        self.channels = int(channels)
        if self.channels < 1:
            raise ValueError("The number of field channels must be positive")
        self.frames = frames_per_clip
        self.temporal_stride, self.spatial_stride = int(temporal_stride), int(spatial_stride)
        self.training, self.random_temporal_crop = training, random_temporal_crop
        if min(self.temporal_stride, self.spatial_stride, trajectory_stride) < 1:
            raise ValueError("Sampling strides must be positive")
        self.records, self.groups, self._handles = [], [], {}
        self.spatial_shape = None
        self.condition_dim = None
        for file_idx, path in enumerate(self.files):
            with h5py.File(path, "r") as handle:
                candidates = [k for k, v in handle.items()
                              if isinstance(v, h5py.Group) and states_key in v]
                if group_name == "auto" and len(candidates) != 1:
                    raise ValueError(f"Expected one trajectory group in {path}")
                group = candidates[0] if group_name == "auto" else group_name
                shape = handle[group][states_key].shape
                if len(shape) not in (4, 5) or shape[1] != self.channels:
                    raise ValueError(f"Expected [N,{channels},H,W,T] or [N,{channels},X,T], got {shape}")
                spatial = shape[2:-1] if len(shape) == 5 else (1, shape[2])
                spatial = tuple((size + self.spatial_stride - 1) // self.spatial_stride for size in spatial)
                if self.spatial_shape is not None and self.spatial_shape != spatial:
                    raise ValueError("All trajectory files must have the same spatial shape")
                self.spatial_shape = spatial
                if self.frames and (shape[-1] + temporal_stride - 1) // temporal_stride < self.frames:
                    raise ValueError(f"Too few frames in {path}")
                for key in self.condition_keys:
                    if handle[group][key].shape[0] != shape[0]:
                        raise ValueError("State and parameter counts differ")
                if shape[0]:
                    parameter_dim = self._parameters(handle[group], 0).shape[-1]
                    if self.condition_dim is not None and self.condition_dim != parameter_dim:
                        raise ValueError("All trajectory files must have the same parameter dimension")
                    self.condition_dim = parameter_dim
                self.groups.append(group)
                self.records.extend((file_idx, i) for i in range(0, shape[0], trajectory_stride))
        if record_indices is not None:
            self.records = [self.records[int(i)] for i in record_indices]
        if not self.records:
            raise ValueError("Empty trajectory selection")
        if condition_dim is not None and int(condition_dim) != self.condition_dim:
            raise ValueError("Configured condition_dim differs from the HDF5 parameters")
        if (channel_mean is None) != (channel_std is None):
            raise ValueError("Specify both channel_mean and channel_std")
        self.mean = None if channel_mean is None else torch.tensor(channel_mean).float().view(-1, 1, 1, 1)
        self.std = None if channel_std is None else torch.tensor(channel_std).float().view(-1, 1, 1, 1)
        if self.std is not None and (self.std.numel() != channels or self.mean.numel() != channels
                                     or not torch.isfinite(self.std).all()
                                     or not torch.isfinite(self.mean).all() or torch.any(self.std <= 0)):
            raise ValueError("Each channel requires a finite mean and positive standard deviation")

    def _parameters(self, group, index):
        values = [np.asarray(group[key][index], dtype=np.float32) for key in self.condition_keys]
        if not values:
            return np.zeros(1, dtype=np.float32)
        values = [value.reshape(1) if value.ndim == 0 else value for value in values]
        if any(value.ndim not in (1, 2) for value in values):
            raise ValueError("Parameters must be vectors or transition-by-parameter matrices")
        timed = [value.shape[0] for value in values if value.ndim == 2]
        if timed:
            if any(length != timed[0] for length in timed):
                raise ValueError("Time-dependent parameter arrays have different transition counts")
            values = [np.broadcast_to(value, (timed[0], value.shape[-1]))
                      if value.ndim == 1 else value for value in values]
        parameters = np.concatenate(values, axis=-1)
        if parameters.shape[-1] < 1:
            raise ValueError("At least one condition coordinate is required")
        if not np.isfinite(parameters).all():
            raise ValueError("Nonfinite PDE parameters")
        return parameters

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __getitem__(self, index):
        file_idx, record_idx = self.records[index]
        if file_idx not in self._handles:
            self._handles[file_idx] = h5py.File(self.files[file_idx], "r")
        group = self._handles[file_idx][self.groups[file_idx]]
        field = group[self.states_key]
        if field.ndim == 5:
            states = np.asarray(field[record_idx, :, ::self.spatial_stride,
                                      ::self.spatial_stride, ::self.temporal_stride], dtype=np.float32)
        else:
            states = np.asarray(field[record_idx, :, ::self.spatial_stride,
                                      ::self.temporal_stride], dtype=np.float32)[:, None]
        frames = self.frames or states.shape[-1]
        max_start = states.shape[-1] - frames
        start = int(torch.randint(max_start + 1, ()).item()) if (
            self.training and self.random_temporal_crop and max_start > 0) else 0
        video = torch.from_numpy(np.ascontiguousarray(states[..., start:start + frames].transpose(0, 3, 1, 2)))
        if self.mean is not None:
            video = (video - self.mean) / self.std
        parameters = self._parameters(group, record_idx)
        if parameters.ndim == 2:
            if self.random_temporal_crop or parameters.shape[0] != frames - 1:
                raise ValueError("Transition parameters must match the selected frames without random temporal cropping")
        indices = [np.arange(start, start + frames, dtype=np.int64) * self.temporal_stride]
        return [video], torch.from_numpy(np.array(parameters, copy=True)), indices


def build_dataset(data, source, **overrides):
    """Build the shared trajectory loader from ``data.dataset`` settings.

    Common dataset options may also be placed directly under ``data``.
    Explicit keyword arguments take precedence. ``frames`` or a single
    ``dataset_fpcs`` entry supplies the requested number of frames.
    """
    options = dict(data.get("dataset", data.get("zebra_cfd", {})) or {})
    for key in ("group_name", "states_key", "condition_key", "condition_dim", "channels",
                "channel_mean", "channel_std", "temporal_stride", "spatial_stride",
                "trajectory_stride", "record_indices", "random_temporal_crop", "file_pattern"):
        if key in data:
            options[key] = data[key]
    if "frames" in data:
        options["frames_per_clip"] = data["frames"]
    elif "dataset_fpcs" in data:
        counts = data["dataset_fpcs"]
        if len(counts) != 1:
            raise ValueError("A PDE trajectory loader requires exactly one frame count")
        options["frames_per_clip"] = counts[0]
    options.update(overrides)
    return PDETrajectoryDataset(source, **options)
