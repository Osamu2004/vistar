# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional
from apps.builder import  make_model
import torch
import torch.nn as nn

from apps.registry import MODEL
from .base import BaseClassifier


@MODEL.register("ImageClassifier")
class ImageClassifier(BaseClassifier):
    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 init_cfg: Optional[dict] = None):
        if pretrained is not None:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        super(ImageClassifier, self).__init__(
            init_cfg=init_cfg)

        if not isinstance(backbone, nn.Module):
            backbone = make_model(backbone)
        if neck is not None and not isinstance(neck, nn.Module):
            neck = make_model(neck)
        if head is not None and not isinstance(head, nn.Module):
            head = make_model(head)

        self.backbone = backbone
        self.neck = neck
        self.head = head


    def forward(self,
                samples,
                mode: str = 'tensor'):
        if mode == 'tensor':
            feats = self.backbone(samples["image"])
            if self.with_neck:
                feats = self.neck(feats)
            if self.with_head:
                feats = self.head(feats)
            return feats
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')
