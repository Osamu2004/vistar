""" 'Fast' Normalization Functions

For GroupNorm and LayerNorm these functions bypass typical AMP upcast to float32.

Additionally, for LayerNorm, the APEX fused LN is used if available (which also does not upcast)

Hacked together by / Copyright 2022 Ross Wightman
"""
try:
    from apex.normalization.fused_layer_norm import fused_layer_norm_affine
    has_apex = True
except ImportError:
    has_apex = False

from typing import List, Optional

import torch
from torch.nn import functional as F
from .config import is_autocast_enabled, get_autocast_dtype


def fast_layer_norm(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5
) -> torch.Tensor:
    if torch.jit.is_scripting():
        # currently cannot use is_autocast_enabled within torchscript
        return F.layer_norm(x, normalized_shape, weight, bias, eps)

    if has_apex:
        return fused_layer_norm_affine(x, weight, bias, normalized_shape, eps)

    if is_autocast_enabled(x.device.type):
        # normally native AMP casts LN inputs to float32
        # apex LN does not, this is behaving like Apex
        dt = get_autocast_dtype(x.device.type)
        x, weight, bias = x.to(dt), weight.to(dt), bias.to(dt) if bias is not None else None

    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        return F.layer_norm(x, normalized_shape, weight, bias, eps)


