import torch.nn as nn
from abc import ABCMeta
from typing import Iterable, List, Optional, Union
import copy

class BaseModule(nn.Module, metaclass=ABCMeta):
    def __init__(self, init_cfg: Union[dict, List[dict], None] = None):
        super().__init__()
        # define default value of init_cfg instead of hard code
        # in init_weights() function
        self._is_init = False

        self.init_cfg = copy.deepcopy(init_cfg)