# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from multiprocessing import Value

import torch

from pde_jepa.models.patch_embed import spatial_pair


class MaskCollator(object):

    def __init__(
        self,
        cfgs_mask,
        dataset_fpcs,
        crop_size=(224, 224),
        patch_size=(16, 16),
        tubelet_size=2,
    ):
        super(MaskCollator, self).__init__()

        self.mask_generators = dict()
        for fpc in dataset_fpcs:
            self.mask_generators[fpc] = []
            for m in cfgs_mask:
                mask_generator = _MaskGenerator(
                    crop_size=crop_size,
                    num_frames=fpc,
                    spatial_patch_size=patch_size,
                    temporal_patch_size=tubelet_size,
                    spatial_pred_mask_scale=m.get("spatial_scale"),
                    temporal_pred_mask_scale=m.get("temporal_scale"),
                    aspect_ratio=m.get("aspect_ratio"),
                    npred=m.get("num_blocks"),
                    max_context_frames_ratio=m.get("max_temporal_keep", 1.0),
                    max_keep=m.get("max_keep", None),
                )
                self.mask_generators[fpc].append(mask_generator)

    def step(self):
        for fpc in self.mask_generators:
            for mask_generator in self.mask_generators[fpc]:
                mask_generator.step()

    def __call__(self, batch):

        filtered_batches = {fpc: [] for fpc in self.mask_generators}
        for sample in batch:
            fpc = len(sample[-1][-1])
            if fpc in filtered_batches:
                filtered_batches[fpc] += [sample]

        fpc_collations = []
        for fpc in filtered_batches:
            fpc_batch = filtered_batches[fpc]
            batch_size = len(fpc_batch)
            if batch_size == 0:
                continue
            collated_batch = torch.utils.data.default_collate(fpc_batch)
            collated_masks_pred, collated_masks_enc = [], []
            for i, mask_generator in enumerate(self.mask_generators[fpc]):
                masks_enc, masks_pred = mask_generator(batch_size)
                collated_masks_enc.append(masks_enc)
                collated_masks_pred.append(masks_pred)
            fpc_collations += [
                (collated_batch, collated_masks_enc, collated_masks_pred)
            ]

        return fpc_collations

class _MaskGenerator(object):

    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        spatial_pred_mask_scale=(0.2, 0.8),
        temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.3, 3.0),
        npred=1,
        max_context_frames_ratio=1.0,
        max_keep=None,
    ):
        super(_MaskGenerator, self).__init__()
        crop_size = spatial_pair(crop_size)
        spatial_patch_size = spatial_pair(spatial_patch_size)
        self.crop_size = crop_size
        self.height, self.width = [
            crop_size[i] // spatial_patch_size[i] for i in (0, 1)
        ]
        if not isinstance(num_frames, int) or not isinstance(temporal_patch_size, int) or min(num_frames, temporal_patch_size) < 1:
            raise ValueError("Frame and temporal patch counts must be positive integers")
        self.duration = num_frames // temporal_patch_size
        if min(self.height, self.width, self.duration) < 1:
            raise ValueError("Patch dimensions must fit within the video")
        if self.duration * self.height * self.width < 2:
            raise ValueError("Masking needs at least two tokens for context and prediction")
        if not isinstance(npred, int) or npred < 1:
            raise ValueError("The number of prediction blocks must be a positive integer")
        if max_keep is not None and (not isinstance(max_keep, int) or max_keep < 1):
            raise ValueError("max_keep must retain at least one context token")
        if not math.isfinite(max_context_frames_ratio) or not 0 < max_context_frames_ratio <= 1:
            raise ValueError("max_context_frames_ratio must be in (0,1]")
        for name, interval, upper in (
            ("spatial scale", spatial_pred_mask_scale, 1.0),
            ("temporal scale", temporal_pred_mask_scale, 1.0),
            ("aspect ratio", aspect_ratio, math.inf),
        ):
            if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                    or not all(math.isfinite(value) for value in interval)
                    or not 0 < interval[0] <= interval[1] <= upper):
                raise ValueError(f"Invalid {name} interval")

        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.npred = npred
        self.max_context_duration = max(
            1, int(self.duration * max_context_frames_ratio)
        )  # maximum number of time-steps (frames) spanned by context mask
        self.max_keep = max_keep  # maximum number of patches to keep in context
        self._itr_counter = Value("i", -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(
        self, generator, temporal_scale, spatial_scale, aspect_ratio_scale
    ):
        # -- Sample temporal block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_t, max_t = temporal_scale
        temporal_mask_scale = min_t + _rand * (max_t - min_t)
        t = max(1, int(self.duration * temporal_mask_scale))

        # -- Sample spatial block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = spatial_scale
        spatial_mask_scale = min_s + _rand * (max_s - min_s)
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)

        # -- Sample block aspect-ratio
        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # -- Compute block height and width (given scale and aspect-ratio)
        if self.height == 1:
            h, w = 1, max(1, min(spatial_num_keep, self.width))
        elif self.width == 1:
            h, w = max(1, min(spatial_num_keep, self.height)), 1
        else:
            h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
            w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
            h = min(h, self.height)
            w = min(w, self.width)

        return (t, h, w)

    def _sample_block_mask(self, b_size):
        t, h, w = b_size
        top = torch.randint(0, self.height - h + 1, (1,))
        left = torch.randint(0, self.width - w + 1, (1,))
        start = torch.randint(0, self.duration - t + 1, (1,))

        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        mask[start : start + t, top : top + h, left : left + w] = 0

        # Context mask will only span the first X frames
        # (X=self.max_context_frames)
        if self.max_context_duration < self.duration:
            mask[self.max_context_duration :, :, :] = 0

        # --
        return mask

    def __call__(self, batch_size):
        """
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample pred block size using seed
        # 2. sample several pred block locations for each image (w/o seed)
        # 3. return pred masks and complement (enc mask)
        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Mask batch size must be a positive integer")
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            temporal_scale=self.temporal_pred_mask_scale,
            spatial_scale=self.spatial_pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )
        if min(p_size) < 1:
            raise ValueError("Sampled prediction block is empty; increase the mask scale")
        if p_size == (self.duration, self.height, self.width):
            raise ValueError("Prediction block covers every token; reduce the mask scale")

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_enc = min_keep_pred = self.duration * self.height * self.width
        for _ in range(batch_size):

            empty_context = True
            attempts = 0
            while empty_context:
                attempts += 1
                if attempts > 256:
                    raise ValueError("Unable to retain context tokens; reduce the mask scale or block count")

                mask_e = torch.ones(
                    (self.duration, self.height, self.width), dtype=torch.int32
                )
                for _ in range(self.npred):
                    mask_e *= self._sample_block_mask(p_size)
                mask_e = mask_e.flatten()

                mask_p = torch.argwhere(mask_e == 0).flatten()
                mask_e = torch.nonzero(mask_e).flatten()

                empty_context = len(mask_e) == 0
                if not empty_context:
                    min_keep_pred = min(min_keep_pred, len(mask_p))
                    min_keep_enc = min(min_keep_enc, len(mask_e))
                    collated_masks_pred.append(mask_p)
                    collated_masks_enc.append(mask_e)

        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
        collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        return collated_masks_enc, collated_masks_pred

def apply_masks(x, masks, concat=True):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors of shape [B, K] containing indices of K patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x

    return torch.cat(all_x, dim=0)
