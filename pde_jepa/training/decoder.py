"""Train the field decoder on frozen PSP rollouts with relative L2 loss."""

import json
import os
from pathlib import Path
import random

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from pde_jepa.data import build_dataset
from pde_jepa.metrics import relative_l2, trajectory_metrics
from pde_jepa.models.decoder import FieldDecoder
from pde_jepa.rollout import FrozenRollout, autocast, decoder_tokens, resolve_dtype
from pde_jepa.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule


def dataset(config, split):
    data = config["data"]
    return build_dataset(data, data[split], training=split == "train", random_temporal_crop=False)


def target_fields(video, config):
    channels = config['data'].get('target_channels')
    if channels is not None:
        if not channels or any(not isinstance(i, int) or i < 0 or i >= video.shape[1] for i in channels):
            raise ValueError('target_channels must select valid physical field channels')
        video = video[:, channels]
    space = config.get('decoder_output_space', 'normalized')
    if space == 'physical':
        mean, std = channel_statistics(config)
        shape = (1, -1) + (1,) * (video.ndim - 2)
        scale = torch.as_tensor(std, device=video.device, dtype=torch.float32).reshape(shape)
        center = torch.as_tensor(mean, device=video.device, dtype=torch.float32).reshape(shape)
        video = video.float() * scale + center
    elif space != 'normalized':
        raise ValueError('decoder_output_space must be normalized or physical')
    return video


def predictor_digest(checkpoint):
    cache = checkpoint.get('structured_ode_rollout_cache') or checkpoint.get('rollout_cache') or {}
    return cache.get('generator_state_sha256', checkpoint.get('predictor_authority', {}).get('predictor_state_sha256'))


def loader(config, data, training=False, batch_size=None):
    sampler = DistributedSampler(data, num_replicas=1, rank=0, shuffle=True) if training else None
    workers = int(config["data"].get("num_workers", 4))
    return DataLoader(data, batch_size=int(batch_size or config["data"]["batch_size"]), sampler=sampler,
                      num_workers=workers, pin_memory=str(config["device"]).startswith("cuda"),
                      persistent_workers=workers > 0, drop_last=training)


def cache_path(config, split):
    return Path(config["cache_dir"]) / f"{split}.h5"


def prepare_caches(config):
    pipeline = FrozenRollout(config, config["device"])
    for split in ("train", "val", "ood"):
        if not config["data"].get(split):
            continue
        ds = dataset(config, split)
        path = cache_path(config, split)
        sources = []
        for filename in ds.files:
            source = Path(filename).resolve()
            status = source.stat()
            sources.append({"path": str(source), "size": status.st_size, "mtime_ns": status.st_mtime_ns})
        provenance = {**pipeline.provenance, "sources": sources,
                      "data": config["data"],
                      "shape": [len(ds), pipeline.frames, pipeline.num_tokens, pipeline.latent_dim]}
        metadata = json.dumps(provenance, sort_keys=True)
        if path.exists():
            with h5py.File(path, "r") as handle:
                if handle.attrs.get("complete", False) and handle.attrs.get("provenance") == metadata:
                    continue
            raise ValueError(f"Incomplete or stale rollout cache: {path}. Use a new cache_dir.")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial.h5")
        with h5py.File(temporary, "a") as handle:
            if "q" in handle:
                if handle.attrs.get("provenance") != metadata or list(handle["q"].shape) != provenance["shape"]:
                    raise ValueError(f"Partial cache provenance differs: {temporary}")
                q, offset = handle["q"], int(handle.attrs["completed_rows"])
            else:
                storage_dtype = 'uint16' if pipeline.storage_dtype == torch.bfloat16 else str(pipeline.storage_dtype).split('.')[-1]
                q = handle.create_dataset("q", shape=tuple(provenance["shape"]), dtype=storage_dtype,
                                          chunks=(1, 1, pipeline.num_tokens, pipeline.latent_dim))
                handle.attrs["provenance"] = metadata
                handle.attrs["completed_rows"] = 0
                offset = 0
            remaining = torch.utils.data.Subset(ds, range(offset, len(ds)))
            for videos, parameters, _ in loader(config, remaining, batch_size=config['data'].get('rollout_batch_size')):
                video = videos[0].to(pipeline.device)
                states = pipeline(video, parameters.to(pipeline.device))
                count = len(video)
                stored = states.cpu().contiguous()
                if stored.dtype == torch.bfloat16:
                    stored = stored.view(torch.uint16)
                q[offset:offset + count] = stored.numpy()
                handle.flush()
                offset += count
                handle.attrs["completed_rows"] = offset
                handle.flush()
                print(f"rollout cache {split}: {offset}/{len(ds)}", flush=True)
            if offset != len(ds):
                raise RuntimeError("Incomplete rollout extraction")
            handle.attrs["complete"] = True
        os.replace(temporary, path)


class RolloutDataset(Dataset):
    """Frozen latent trajectories paired with their physical targets."""

    def __init__(self, config, split):
        self.physical = dataset(config, split)
        self.path = cache_path(config, split)
        with h5py.File(self.path, "r") as handle:
            if not handle.attrs.get("complete", False) or len(handle["q"]) != len(self.physical):
                raise ValueError(f"Incomplete rollout cache {self.path}")
            self.provenance = json.loads(handle.attrs["provenance"])
        self._handle = None

    def __len__(self):
        return len(self.physical)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __getitem__(self, index):
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        stored = np.asarray(self._handle["q"][index]).copy()
        q = torch.from_numpy(stored)
        if stored.dtype == np.uint16:
            q = q.view(torch.bfloat16)
        return q, self.physical[index][0][0]


def build_decoder(config, checkpoint=None):
    saved = {} if checkpoint is None else checkpoint.get('model_config', {}).get('decoder', {})
    model = dict(saved or config['model'])
    model.pop("task", None)
    family = model.pop('family', config['model'].get('family', 'field'))
    pretrain = yaml.safe_load(Path(config["representation"]["encoder_pretrain_config"]).read_text())
    data = pretrain["data"]
    fields = {**(data.get("dataset", data.get("zebra_cfd", {})) or {}), **data}
    output_data = config["data"]
    output_fields = {**(output_data.get("dataset", output_data.get("zebra_cfd", {})) or {}), **output_data}
    encoder = pretrain["model"]
    target_channels = output_data.get("target_channels")
    defaults = dict(img_size=data["crop_size"], patch_size=data["patch_size"],
                    num_frames=int(config["data"]["frames"]), tubelet_size=1,
                    encoder_embed_dim=encoder.get("embed_dim", 192),
                    out_chans=(len(target_channels) if target_channels is not None
                               else output_fields.get("channels", fields.get("channels", 1))))
    if family in ('progressive_1d', 'low_frequency_1d'):
        from pde_jepa.models.sequence_1d import ProgressivePDEDecoder1D, ProgressiveLowFrequencyPDEDecoder1D
        height, width = data['crop_size']
        patch_height, patch_width = data['patch_size']
        if height != 1 or patch_height != 1 or defaults['out_chans'] != 1:
            raise ValueError('One-dimensional decoders require a single field on a [1,X] grid')
        cls = ProgressivePDEDecoder1D if family == 'progressive_1d' else ProgressiveLowFrequencyPDEDecoder1D
        decoder = cls(**{**dict(latent_dim=defaults['encoder_embed_dim'], tokens=width // patch_width,
                              output_points=width, output_frames=1), **model})
    elif family == 'field':
        decoder = FieldDecoder(**{**defaults, **model})
    else:
        raise ValueError(f'Unknown decoder family {family!r}')
    if checkpoint is not None:
        decoder.load_state_dict(checkpoint["decoder"], strict=True)
    return decoder.to(config["device"])


def decode_fields(decoder, q, device, config):
    dtype = resolve_dtype(config.get('precision', {}).get('decoder', 'bfloat16'))
    normalize = bool(config.get('normalize_representations', True))
    with autocast(torch.device(device), dtype):
        tokens = decoder_tokens(q, device, normalize=normalize)
        if dtype == torch.float32:
            tokens = tokens.float()
        if hasattr(decoder, 'output_points'):
            batch, frames = q.shape[:2]
            input_dtype = config.get('precision', {}).get('decoder_input', 'float32')
            if input_dtype != 'native':
                tokens = tokens.to(resolve_dtype(input_dtype))
            result = decoder(tokens.reshape(batch * frames, decoder.tokens, decoder.latent_dim))
            return result.reshape(batch, 1, frames, 1, decoder.output_points)
        return decoder(tokens)


@torch.inference_mode()
def evaluate_cached(decoder, data_loader, device, config, observed_frames=1):
    decoder.eval()
    loss_sum, count, totals = 0.0, 0, {}
    for q, video in data_loader:
        q, video = q.to(device), video.to(device)
        video = target_fields(video, config)
        prediction = decode_fields(decoder, q, device, config)
        loss_sum += relative_l2(prediction, video, reduction="none").double().sum().item()
        for key, value in physical_metrics(prediction, video, config, observed_frames).items():
            totals[key] = totals.get(key, 0.0) + value.double().sum().item()
        count += len(video)
    return {"loss": loss_sum / count, "num_samples": count,
            **{key: value / count for key, value in totals.items()}}


def channel_statistics(config):
    data = config["data"]
    fields = {**(data.get("dataset", data.get("zebra_cfd", {})) or {}), **data}
    mean, std = fields.get("channel_mean"), fields.get("channel_std")
    channels = data.get("target_channels")
    if channels is not None:
        mean = [mean[i] for i in channels] if mean is not None else None
        std = [std[i] for i in channels] if std is not None else None
    return (0.0 if mean is None else mean, 1.0 if std is None else std)


def physical_metrics(prediction, target, config, observed_frames=1):
    mean, std = channel_statistics(config) if config.get('decoder_output_space', 'normalized') == 'normalized' else (0.0, 1.0)
    return trajectory_metrics(prediction, target,
                              channel_mean=mean, channel_std=std,
                              observed_frames=observed_frames)


def save_checkpoint(path, payload):
    temporary = str(path) + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(config):
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("Decoder training requires a single process")
    seed = int(config.get("seed", 239))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(config["device"])
    prepare_caches(config)
    train_data = RolloutDataset(config, "train")
    train_loader = loader(config, train_data, training=True)
    validation = {s: loader(config, RolloutDataset(config, s)) for s in ("val", "ood")
                  if config["data"].get(s)}
    # Cache creation must not change the decoder's initialization or sampling RNG.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    decoder = build_decoder(config)
    optim = config["optimization"]
    decay, no_decay = [], []
    for name, parameter in decoder.named_parameters():
        (no_decay if name.endswith("bias") or parameter.ndim == 1 else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay}, {"params": no_decay, "weight_decay": 0.0, "WD_exclude": True}],
        betas=tuple(optim["betas"]), eps=float(optim["eps"]),
    )
    total_steps = len(train_loader) * int(optim["epochs"])
    lr_schedule = WarmupCosineSchedule(optimizer, warmup_steps=int(optim["warmup"] * len(train_loader)),
                                      start_lr=optim["start_lr"], ref_lr=optim["lr"],
                                      final_lr=optim["final_lr"], T_max=total_steps)
    wd_schedule = CosineWDSchedule(optimizer, ref_wd=optim["weight_decay"],
                                   final_wd=optim["weight_decay"], T_max=total_steps)
    folder = Path(config["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    start, step, best, best_epoch = 0, 0, float("inf"), 0
    if config.get("resume"):
        ck = torch.load(config["resume"], map_location="cpu", weights_only=False)
        if predictor_digest(ck) != train_data.provenance["psp_sha256"]:
            raise ValueError("Cannot resume decoder training with a different PSP checkpoint")
        decoder.load_state_dict(ck["decoder"], strict=True)
        optimizer.load_state_dict(ck["opt"])
        start, step = ck["epoch"], ck["global_step"]
        best, best_epoch = ck["best_val_loss"], ck["best_epoch"]
        for _ in range(step):
            lr_schedule.step()
            wd_schedule.step()
    for epoch in range(start, int(optim["epochs"])):
        decoder.train()
        train_loader.sampler.set_epoch(epoch)
        loss_total = 0.0
        for q, video in train_loader:
            lr_schedule.step()
            wd_schedule.step()
            optimizer.zero_grad(set_to_none=True)
            q, video = q.to(device), video.to(device)
            video = target_fields(video, config)
            prediction = decode_fields(decoder, q, device, config)
            loss = relative_l2(prediction, video)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite decoder loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), float(optim["clip_grad"]))
            optimizer.step()
            loss_total += loss.detach().item()
            step += 1
        observed = int(train_data.provenance.get("observed_frames", 1))
        metrics = {s: evaluate_cached(decoder, dl, device, config, observed) for s, dl in validation.items()}
        improved = metrics["val"]["loss"] < best
        if improved:
            best, best_epoch = metrics["val"]["loss"], epoch + 1
        payload = {"decoder": decoder.state_dict(), "opt": optimizer.state_dict(),
                   "epoch": epoch + 1, "global_step": step, "best_val_loss": best, "best_epoch": best_epoch,
                   "model_config": {"decoder": config["model"]}, "data_config": config["data"],
                   "val_nrmse": metrics["val"]["all_frame_nrmse"],
                   "structured_ode_rollout_cache": {"generator_state_sha256": train_data.provenance["psp_sha256"],
                                                     "normalize_representations": bool(config.get('normalize_representations', True))},
                   "metrics": metrics, "config": config}
        save_checkpoint(folder / "latest.pt", payload)
        if improved:
            save_checkpoint(folder / "best.pt", payload)
        if (epoch + 1) % int(config.get("save_every", 10)) == 0:
            save_checkpoint(folder / f"epoch{epoch + 1:04d}.pt", payload)
        row = {"epoch": epoch + 1, "train_loss": loss_total / len(train_loader), **metrics}
        print(json.dumps(row), flush=True)
        with (folder / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")


@torch.inference_mode()
def evaluate_checkpoint(config, checkpoint_path, split="ood", limit=None):
    """The training validation loop, generating rollouts directly when caches are absent."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pipeline = FrozenRollout(config, config["device"])
    expected = predictor_digest(checkpoint) or config.get('psp_sha256')
    if expected is None:
        raise ValueError('Decoder checkpoint must identify its PSP weights or set psp_sha256 explicitly')
    if pipeline.provenance["psp_sha256"] != expected:
        raise ValueError("PSP checkpoint differs from the decoder's training predictor")
    decoder = build_decoder(config, checkpoint).eval().requires_grad_(False)
    ds = dataset(config, split)
    if limit is not None:
        if limit < 1:
            raise ValueError("Evaluation limit must be positive")
        ds = torch.utils.data.Subset(ds, range(min(limit, len(ds))))
    totals, count = {}, 0
    for videos, parameters, _ in loader(config, ds, batch_size=config['data'].get('rollout_batch_size')):
        video, parameters = videos[0].to(pipeline.device), parameters.to(pipeline.device)
        states = pipeline(video, parameters)
        video = target_fields(video, config)
        decoder_batch_size = int(config['data'].get('decoder_batch_size', config['data']['batch_size']))
        if decoder_batch_size < 1:
            raise ValueError('decoder_batch_size must be positive')
        for q, target in zip(states.split(decoder_batch_size), video.split(decoder_batch_size)):
            prediction = decode_fields(decoder, q, pipeline.device, config)
            for key, value in physical_metrics(prediction, target, config, pipeline.observed_frames).items():
                totals[key] = totals.get(key, 0.0) + value.double().sum().item()
        count += len(video)
        print(f"{split}: {count}/{len(ds)}", flush=True)
    result = {"checkpoint": str(Path(checkpoint_path).resolve()), "decoder_epoch": checkpoint["epoch"],
              "split": split, "num_samples": count, "observed_frames": pipeline.observed_frames,
              "rollout_steps": pipeline.frames - pipeline.observed_frames,
              "provenance": pipeline.provenance, **{k: v / count for k, v in totals.items()}}
    folder = Path(config["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"evaluation_{split}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result
