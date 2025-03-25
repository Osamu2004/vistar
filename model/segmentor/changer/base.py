from model.segmentor.base import BaseSegmentor
import torch
from typing import Dict, Optional


class BaseChanger(BaseSegmentor):
    """Base class for all change detection qsegmentation models."""
    def __init__(self, 
                 init_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None):

        super().__init__(init_cfg=init_cfg,test_cfg=test_cfg)