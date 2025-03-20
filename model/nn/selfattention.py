from typing import List, Optional, Type, Union

import torch
from torch import nn as nn
from torch.nn import functional as F

from .config import use_fused_attn
from .create_conv2d import create_conv2d
from .helpers import to_2tuple
from .pool2d_same import create_pool2d


class MultiQueryAttentionV2(nn.Module):
    """Multi Query Attention.

    Fast Transformer Decoding: One Write-Head is All You Need
    https://arxiv.org/pdf/1911.02150.pdf

    This is an acceletor optimized version - removing multiple unnecessary
    tensor transpose by re-arranging indices according to the following rules: 1)
    contracted indices are at the end, 2) other indices have the same order in the
    input and output tensores.

    Compared to V1, this gives 3x speed up.
    """

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            num_heads: int = 8,
            key_dim: int = 64,
            value_dim: int = 64,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ):
        """Initializer."""
        super().__init__()
        dim_out = dim_out or dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.scale = key_dim ** -0.5

        self.query_proj = nn.Parameter(torch.randn([self.num_heads, self.key_dim, dim]))
        self.key_proj = nn.Parameter(torch.randn([dim, self.key_dim]))
        self.value_proj = nn.Parameter(torch.randn([dim, self.value_dim]))
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Parameter(torch.randn([dim_out, self.num_heads, self.value_dim]))
        self.proj_drop = nn.Dropout(proj_drop)

    def _reshape_input(self, t):
        """Reshapes a tensor to three dimensions, keeping the first and last."""
        s = t.shape
        # Propagate the shape statically where possible.
        #num = t.shape[1:-1].numel()
        #return t.reshape(s[0], num, s[-1])
        return t.reshape(s[0], s[1], -1).transpose(1, 2)

    def forward(self, x, m: Optional[torch.Tensor] = None):
        """Run layer computation."""
        b, _, h, w = x.shape
        m = m if m is not None else x

        reshaped_x = self._reshape_input(x)
        reshaped_m = self._reshape_input(m)

        q = torch.einsum('bnd,hkd->bnhk', reshaped_x, self.query_proj)
        k = torch.einsum('bmd,dk->bmk', reshaped_m, self.key_proj)

        attn = torch.einsum('bnhk,bmk->bnhm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        v = torch.einsum('bmd,dv->bmv', reshaped_m, self.value_proj)
        o = torch.einsum('bnhm,bmv->bnhv', attn, v)
        result = torch.einsum('bnhv,dhv->bdn', o, self.out_proj)
        result = self.proj_drop(result)
        return result.reshape(b, -1, h, w)



class Attention2d(nn.Module):
    fused_attn: torch.jit.Final[bool]

    """ multi-head attention for 2D NCHW tensors"""
    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            num_heads: int = 32,
            bias: bool = True,
            expand_first: bool = False,
            head_first: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.
    ):
        super().__init__()
        dim_out = dim_out or dim
        dim_attn = dim_out if expand_first else dim
        self.num_heads = num_heads
        self.dim_head = dim_attn // num_heads
        self.head_first = head_first
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Conv2d(dim, dim_attn * 3, 1, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim_attn, dim_out, 1, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        B, C, H, W = x.shape

        if self.head_first:
            q, k, v = self.qkv(x).view(B, self.num_heads, self.dim_head * 3, -1).chunk(3, dim=2)
        else:
            q, k, v = self.qkv(x).reshape(B, 3, self.num_heads, self.dim_head, -1).unbind(1)

        if self.fused_attn:
            x = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(-1, -2).contiguous(),
                k.transpose(-1, -2).contiguous(),
                v.transpose(-1, -2).contiguous(),
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            ).transpose(-1, -2).reshape(B, -1, H, W)
        else:
            q = q.transpose(-1, -2)
            v = v.transpose(-1, -2)
            attn = q @ k * q.size(-1) ** -0.5
            if attn_mask is not None:
                # NOTE: assumes mask is float and in correct shape
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(-1, -2).reshape(B, -1, H, W)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x