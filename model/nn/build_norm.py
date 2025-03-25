import torch
import torch.nn as nn
from .utils import build_kwargs_from_config
from timm.layers import LayerNorm2d
from .fast_norm import fast_layer_norm
from .config import is_fast_norm
from .identity import IdentityLayer 
from torch.nn import functional as F
__all__ = [ "build_norm"]


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    _fast_norm: torch.jit.Final[bool]

    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)
        self._fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class LayerNorm(nn.LayerNorm):
    """ LayerNorm w/ fast norm option
    """
    _fast_norm: torch.jit.Final[bool]

    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)
        self._fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x

REGISTERED_NORM_DICT = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
    "ln2d": LayerNorm2d,
}

def build_norm(name="bn2d", num_features=None, **kwargs) -> nn.Module or None:
    if name is None:
        return IdentityLayer()  # 如果name为None，返回IdentityLayer实例
    if name in ["ln"]:
        kwargs["normalized_shape"] = num_features
    if name in [ "ln2d"]:
        kwargs["num_channels"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        raise ValueError(f"Norm type '{name}' is not registered.")
    