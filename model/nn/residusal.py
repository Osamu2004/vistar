import torch
from torch import nn
from .drop import DropPath
from .layer_scale import LayerScale
from .build_act import build_act
class ResidualBlock(nn.Module): 
    def __init__(
        self,
        main: nn.Module or None,
        shortcut: nn.Module or None,
        post_act=None,
        pre_norm: nn.Module or None = None,
        drop_path_prob: float = 0.0,
        layer_scale: nn.Module or None = None,  
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)
        self.layer_scale = layer_scale

        if drop_path_prob>0.0:
            self.drop_path = DropPath(drop_path_prob)
        else:
            self.drop_path = None

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x)
            if self.layer_scale is not None:
                res = self.layer_scale(res)
            if self.drop_path is not None:
                res = self.drop_path(res)
            res = res + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
                
        return res
