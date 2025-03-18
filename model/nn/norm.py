import torch
import torch.nn as nn
from model_test.utils.network import build_kwargs_from_config
from torch.nn.modules.batchnorm import _BatchNorm
from timm.layers import LayerNorm2d

__all__ = [ "build_norm"]



    
REGISTERED_NORM_DICT = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
    "ln2d": LayerNorm2d,
}

def build_norm(name="bn2d", num_features=None, **kwargs) -> nn.Module or None:
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
    