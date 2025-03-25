# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional
from apps.builder import make_model
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import BaseSegmentor

class EncoderDecoder(BaseSegmentor):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck = None,
                 pretrained: Optional[str] = None,
                 init_cfg = None):
        super().__init__(init_cfg=init_cfg)
        self.backbone = make_model(backbone)
        if neck is not None:
            self.neck = make_model(neck)
        self._init_decode_head(decode_head)



    def _init_decode_head(self, decode_head) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = make_model(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def forward(self,input):
        x = self.backbone(input["image"])





