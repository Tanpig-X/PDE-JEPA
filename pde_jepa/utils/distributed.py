"""Single-process and torchrun distributed initialization."""

import os

import torch
import torch.distributed as dist


def init_distributed(device=None):
    """Initialize torchrun's process group and return (world_size, rank)."""
    if dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    use_cuda = torch.cuda.is_available() if device is None else torch.device(device).type == "cuda"
    if use_cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if world_size > 1:
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    return world_size, rank
