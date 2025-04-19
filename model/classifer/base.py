# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Sequence

import torch
from model.base_module import BaseModule


class BaseClassifier(BaseModule, metaclass=ABCMeta):
    def __init__(self,
                 init_cfg: Optional[dict] = None,):
        super(BaseClassifier, self).__init__(init_cfg)

    @property
    def with_neck(self) -> bool:
        """Whether the classifier has a neck."""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self) -> bool:
        """Whether the classifier has a head."""
        return hasattr(self, 'head') and self.head is not None

