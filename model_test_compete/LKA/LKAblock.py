import torch.nn as nn
import torch

from model_test.nn.ops import LinearLayer,ResidualBlock,DualResidualBlock,IdentityLayer,TimeMBConv,LayerScale2d,build_norm,MBConv,ConvLayer
from model_test.nn.act import build_act
from typing import Tuple
from model_test.nn.nas import TFM1,TFM2,TFM3,TFM4,TFM5,TFM6,TFM7,TFM8,TFM9,TFM10,NASNetwork,TFM_3D_11

from torch.nn import Sequential


class AttentionModule1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Conv2d(dim,dim,1)
        self.act1 = build_act("gelu")
        self.attention = Sequential(
            nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=5,stride=1,groups=dim,padding=2,),
            nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=7,stride=1,groups=dim,dilation=3,padding=9,),
            nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=1,stride=1,groups=1),
        )
        self.proj = nn.Conv2d(dim,dim,1)


    def single(self,t):
        t = self.act1(self.qkv(t))
        attn = self.attention(t)
        t = attn*t
        t = self.proj(t)
        return t
    
    def forward(self, t1, t2):
        t1 = self.single(t1)
        t2 = self.single(t2)
        return t1,t2

class AttentionModule2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fusion = TFM_3D_11(dim)
        self.qkv = nn.Conv2d(dim,dim,1)
        self.act1 = build_act("gelu")
        self.attention = Sequential(
            nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=5,stride=1,groups=dim,padding=2,),
            nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=7,stride=1,groups=dim,dilation=3,padding=9,),
            nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=1,stride=1,groups=1),
        )
        self.proj = nn.Conv2d(dim,dim,1)

    def forward(self, t1, t2):
        t1 = self.act1(self.qkv(t1))
        t2 = self.act1(self.qkv(t2))
        fusion = self.fusion((t1,t2))
        attn = self.attention(fusion)
        t1,t2 = attn*t1,attn*t2
        t1 = self.proj(t1)
        t2 = self.proj(t2)

        return t1,t2


class Moe(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=4,
        use_bias=False,
        norm=(None, None, None),
        act_func=("relu6", "relu6", None),
    ):
        super(Moe, self).__init__()


        self.expert1 = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            mid_channels=mid_channels,
            expand_ratio=expand_ratio,
            use_bias=use_bias,
            norm=norm,
            act_func=act_func,
        )
        self.expert2 = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            mid_channels=mid_channels,
            expand_ratio=expand_ratio,
            use_bias=use_bias,
            norm=norm,
            act_func=act_func,
        )
        self.expert3 = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            mid_channels=mid_channels,
            expand_ratio=expand_ratio,
            use_bias=use_bias,
            norm=norm,
            act_func=act_func,
        )
    def forward(self, t1,t2: torch.Tensor) -> torch.Tensor:
        t1_1 = self.expert1(t1)
        t2_1 = self.expert1(t2)
        t1_2 = self.expert2(t1)
        t2_2 = self.expert3(t2)
        return t1_1+t1_2,t2_1+t2_2



class LKABlock1(nn.Module):
    def __init__(
        self,
        dim: int,
        drop_path_prob:float,
        use_layer_scale:bool,
        expand_ratio: float = 4,
        norm="ln2d",
        act_func="gelu",
    ):
        super(LKABlock1, self).__init__()
        self.context_module = DualResidualBlock(
            main = AttentionModule1(
                dim=dim,
            ),
            shortcut = IdentityLayer(),
            pre_norm = build_norm(norm,num_features=dim),
            drop_path_prob = drop_path_prob,
            layer_scale=LayerScale2d(dim=dim) if use_layer_scale else None,
        )
        local_module = Moe(
            in_channels=dim,
            out_channels=dim,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, None),
            act_func=(act_func, act_func, None),
        )
        self.local_module = DualResidualBlock(
            main = local_module, 
            shortcut = IdentityLayer(),
            pre_norm = build_norm(norm,num_features=dim),
            drop_path_prob = drop_path_prob,
            layer_scale=LayerScale2d(dim=dim) if use_layer_scale else None,
            )

    def forward(self, input:Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        input = self.context_module(input)
        input = self.local_module(input)
        return input

class LKABlock2(nn.Module):
    def __init__(
        self,
        dim: int,
        drop_path_prob:float,
        use_layer_scale:bool,
        expand_ratio: float = 4,
        norm="ln2d",
        act_func="gelu",
    ):
        super(LKABlock2, self).__init__()
        self.context_module = DualResidualBlock(
            main = AttentionModule2(
                dim=dim,
            ),
            shortcut = IdentityLayer(),
            pre_norm = build_norm(norm,num_features=dim),
            drop_path_prob = drop_path_prob,
            layer_scale=LayerScale2d(dim=dim) if use_layer_scale else None,
        )
        local_module = Moe(
            in_channels=dim,
            out_channels=dim,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, None),
            act_func=(act_func, act_func, None),
        )
        self.local_module = DualResidualBlock(
            main = local_module, 
            shortcut = IdentityLayer(),
            pre_norm = build_norm(norm,num_features=dim),
            drop_path_prob = drop_path_prob,
            layer_scale=LayerScale2d(dim=dim) if use_layer_scale else None,
            )

    def forward(self, input:Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        input = self.context_module(input)
        input = self.local_module(input)
        return input
    
class LKABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        drop_path_prob:float,
        use_layer_scale:bool,
        expand_ratio: float = 4,
        norm="ln2d",
        act_func="gelu",
    ):
        super(LKABlock, self).__init__()
        self.first =  LKABlock1(
                dim=dim,  expand_ratio=expand_ratio,
                 drop_path_prob=drop_path_prob,  use_layer_scale=use_layer_scale,norm=norm,act_func=act_func)
                # use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value, 
                # )
        self.second =  LKABlock2(
                dim=dim,  expand_ratio=expand_ratio,
                 drop_path_prob=drop_path_prob,  use_layer_scale=use_layer_scale,norm=norm,act_func=act_func)
    def forward(self, input:Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        input = self.first(input)
        input = self.second(input)
        return input