# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license; see LICENSE and THIRD_PARTY_NOTICES.md.

"""Load trusted training checkpoints containing model and optimizer state."""
import torch


def robust_checkpoint_loader(r_path, map_location="cpu"):
    return torch.load(r_path, map_location=map_location, weights_only=False)
