from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_module import BaseModule

class ClsHead(BaseModule):
    def __init__(self,
                 init_cfg: Optional[dict] = None):
        super(ClsHead, self).__init__(init_cfg=init_cfg)

