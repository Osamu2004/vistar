from .base import BaseChanger
from apps.builder import make_model
from typing import List, Optional
from model.base_module import SiameseSequential
class SEncoderSDecoder(BaseChanger):
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
        self._init_decode_head(backbone)
    def _init_backbone(self, backbone) -> None:
        """Initialize ``decode_head``"""

        self.backbone= make_model(backbone)
        if hasattr(self.backbone, 'siamese'):
            self.backbone.siamese = True
            self.backbone = SiameseSequential(self.backbone)
