# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn


class MultiSeqWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim

    def forward(self, x, masks=None, training_mode=False):
        """
        :param x: [list] List of Tensors of different seq lengths
        :param masks: [list] List of Tensors (index: masks for given seq length)
        """
        if masks is None:
            outputs = []
            for x_fpc in x:
                outputs.append(self.backbone(x_fpc, training=training_mode))
            return outputs

        outs = [[] for _ in x]
        for i, (x_fpc, m_fpc) in enumerate(zip(x, masks)):
            for m in m_fpc:
                outs[i] += [self.backbone(x_fpc, masks=m, training=training_mode)]
        return outs


class PredictorMultiSeqWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks_x, masks_y):
        """
        :param x: [list] List of encoder outputs for different seq lengths
        :param masks_x: [list] List of encoder masks
        :param masks_y: [list] List of predictor masks
        """
        outs_pred = [[] for _ in x]
        outs_context = [[] for _ in x]
        for i, (x_fpc, mx_fpc, my_fpc) in enumerate(zip(x, masks_x, masks_y)):
            for xij, mx, my in zip(x_fpc, mx_fpc, my_fpc):
                x_pred, x_context = self.backbone(xij, mx, my, mask_index=i)
                outs_pred[i] += [x_pred]
                outs_context[i] += [x_context]
        return outs_pred, outs_context
