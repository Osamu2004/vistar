from functools import partial

import torch.nn as nn
from .identity import IdentityLayer
from model.nn.utils import build_kwargs_from_config
__all__ = ["build_act"]


REGISTERED_ACT_DICT = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


def build_act(name: str, **kwargs) -> nn.Module or None:
    if name is None:
        return IdentityLayer()  # 如果name为None，返回IdentityLayer实例
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        raise ValueError(f"Act type '{name}' is not registered.")