try:
    import apex
    _HAS_APEX = True
except ImportError:
    _HAS_APEX = False
import warnings

import torch
from torch.nn import functional as F

# fast (ie lower precision LN) can be disabled with this flag if issues crop up
_USE_FAST_NORM = False  # defaulting to False for now
_HAS_FUSED_ATTN = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
def get_autocast_dtype(device: str = 'cuda'):
    try:
        return torch.get_autocast_dtype(device)
    except (AttributeError, TypeError):
        # dispatch to older device specific fns, only covering cuda/cpu devices here
        if device == 'cpu':
            return torch.get_autocast_cpu_dtype()
        else:
            assert device == 'cuda'
            return torch.get_autocast_gpu_dtype()


def is_autocast_enabled(device: str = 'cuda'):
    try:
        return torch.is_autocast_enabled(device)
    except TypeError:
        # dispatch to older device specific fns, only covering cuda/cpu devices here
        if device == 'cpu':
            return torch.is_autocast_cpu_enabled()
        else:
            assert device == 'cuda'
            return torch.is_autocast_enabled()  # defaults cuda (only cuda on older pytorch)
        
def is_fast_norm():
    return _USE_FAST_NORM


def set_fast_norm(enable=True):
    global _USE_FAST_NORM
    _USE_FAST_NORM = enable

def has_apex():
    return _HAS_APEX


def use_fused_attn() -> bool:
    if not _HAS_FUSED_ATTN:
        return False
    return True

def set_fused_attn(enable: bool = True):
    global _USE_FUSED_ATTN
    if not _HAS_FUSED_ATTN:
        warnings.warn('This version of pytorch does not have F.scaled_dot_product_attention, fused_attn flag ignored.')
        return
    elif enable:
        _USE_FUSED_ATTN = 1
    else:
        _USE_FUSED_ATTN = 0