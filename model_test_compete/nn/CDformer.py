import torch.nn as nn
import torch

from model.nn.ops import LinearLayer,ResidualBlock,DualResidualBlock,IdentityLayer,MBConv,LayerScale2d,build_norm,ConvolutionalGLU
from model.nn.build_act import build_act
from typing import Tuple
class LowMixer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., pool_size=2,
        **kwargs, ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim
        
        self.qkv = LinearLayer(in_features = dim , out_features = dim  * 3, use_bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.pool = nn.AvgPool2d(pool_size, stride=pool_size, padding=0, count_include_pad=False) if pool_size > 1 else nn.Identity()
        self.uppool = nn.Upsample(scale_factor=pool_size) if pool_size > 1 else nn.Identity()
        

    def att_fun(self, q, k, v, B, N, C):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(2, 3).reshape(B, C, N)
        return x

    def forward(self, input):
        x1,x2=input
        # B, C, H, W
        B, _, _, _ = x1.shape
        xa1 = self.pool(x1)
        xa1 = xa1.permute(0, 2, 3, 1).view(B, -1, self.dim)
        xa2 = self.pool(x2)
        xa2 = xa2.permute(0, 2, 3, 1).view(B, -1, self.dim)
        B, N, C = xa1.shape
        qkv1 = self.qkv(xa1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv2 = self.qkv(xa2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q1 , k1, v1 = qkv1.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q2 , k2, v2 = qkv2.unbind(0)
        xa1 = self.att_fun(q1, k1, v1, B, N, C)
        xa2 = self.att_fun(q1, k1, v2, B, N, C)
        xa1 = xa1.view(B, C, int(N**0.5), int(N**0.5))#.permute(0, 3, 1, 2)
        
        xa1 = self.uppool(xa1)
        xa2 = xa2.view(B, C, int(N**0.5), int(N**0.5))#.permute(0, 3, 1, 2)
        
        xa2 = self.uppool(xa2)
        return xa1,xa2
    
class HighMixer(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1,
        **kwargs, ):
        super().__init__()
        
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1,groups=dim)
        self.conv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2,groups=dim)
        self.conv7x7 = nn.Conv2d(dim, dim, kernel_size=7, padding=3,groups=dim)
        self.conv9x9 = nn.Conv2d(dim, dim, kernel_size=9, padding=4,groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, kernel_size=1, stride=stride, bias=False, groups=dim)
        self.mid_gelu1 = build_act("gelu")
        self.Maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.proj2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.mid_gelu2 = build_act("gelu")

    def forward(self, x):
        cx =  x + self.conv3x3(x) +self.conv5x5(x) + self.conv7x7(x) + self.conv9x9(x)
        cx = self.proj1(cx)
        cx = self.mid_gelu1(cx)
        px = self.Maxpool(x)
        px = self.proj2(px)
        px = self.mid_gelu2(px)
        hx = torch.cat((cx, px), dim=1)
        return hx
    
class AttentionModule(nn.Module):
    def __init__(self, dim,num_heads=8, qkv_bias=False, pool_size=2, attention_heads=1,proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.low_dim = low_dim = attention_heads * head_dim
        self.high_dim = high_dim = dim - low_dim
        self.high_mixer = HighMixer(high_dim)
        self.low_mixer = LowMixer(self.low_dim , num_heads=attention_heads, qkv_bias=qkv_bias, attn_drop=0.1, pool_size=pool_size,)
        self.act = nn.GELU()
        self.conv_fuse = nn.Conv2d(low_dim+high_dim*2, low_dim+high_dim*2, kernel_size=3, stride=1, padding=1, bias=False, groups=low_dim+high_dim*2)
        self.proj = nn.Conv2d(low_dim+high_dim*2, dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)

    def single(self,input):
        t1, t2 = input
        hx1 = t1[:,:self.high_dim, :, :].contiguous()
        lx1 = t1[:,self.high_dim:, :, :].contiguous()
        hx2 = t2[:,:self.high_dim, :, :].contiguous()
        lx2 = t2[:,self.high_dim:, :, :].contiguous()

        hx1 = self.high_mixer(hx1)
        hx2 = self.high_mixer(hx2)

        lx1_fused,lx2_fused = self.low_mixer((lx1, lx2))
        attn1 = torch.cat((hx1, lx1_fused), dim=1)
        attn2 = torch.cat((hx2, lx2_fused), dim=1)
        attn1 = attn1 + self.conv_fuse(attn1)
        attn1 = self.proj(attn1)

        attn2 = attn2 + self.conv_fuse(attn2)
        attn2 = self.proj(attn2)  
        return attn1, attn2
    
    def forward(self, t1, t2):
        attn1,attn2 = self.single((t1,t2))
        return attn1,attn2
        
class CBFBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads:int,
        attention_heads:int,
        qkv_bias:int,
        pool_size:int,
        drop_path_prob:float,
        use_layer_scale:bool,
        expand_ratio: float = 4,
        norm="ln2d",
        act_func="gelu",
    ):
        super(CBFBlock, self).__init__()
        self.context_module = DualResidualBlock(
            main = AttentionModule(
                dim=dim,
                num_heads=num_heads,
                attention_heads =attention_heads,
                qkv_bias=qkv_bias,
                pool_size=pool_size,
            ),
            shortcut = IdentityLayer(),
            pre_norm = build_norm(norm,num_features=dim),
            drop_path_prob = drop_path_prob,
            layer_scale=LayerScale2d(dim=dim) if use_layer_scale else None,
        )
        local_module = MBConv(
            in_channels=dim,
            out_channels=dim,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, None),
            act_func=(act_func, act_func, None),
        )
        self.local_module = ResidualBlock(
            main = local_module, 
            shortcut = IdentityLayer(),
            pre_norm = build_norm(norm,num_features=dim),
            drop_path_prob = drop_path_prob,
            layer_scale=LayerScale2d(dim=dim) if use_layer_scale else None,
            )

    def forward(self, input:Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x1,x2 = self.context_module(input)
        x1 = self.local_module(x1)
        x2 = self.local_module(x2)
        return [x1,x2]
    
