# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn

from apps.registry import MODEL
from .base import ClsHead


@MODEL.register('LinearClsHead')
class LinearClsHead(ClsHead):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super(LinearClsHead, self).__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)


    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        # The final classification head.
        cls_score = self.fc(feats)
        return cls_score