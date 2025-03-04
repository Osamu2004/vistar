import torch.nn as nn
import torch

from model.nn.ops import LinearLayer,ResidualBlock,DualResidualBlock,IdentityLayer,MBConv,LayerScale2d,build_norm,ConvolutionalGLU
from model.nn.act import build_act
from typing import Tuple
class LowMixer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.,pool_size=2, proj_drop=0.,
                 agent_num=64, window=14, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.agent_num = agent_num

        #self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        #self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        #self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
        #self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))
        #self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
        #self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))
        #self.ac_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1))
        #self.ca_bias = nn.Parameter(torch.zeros(1, num_heads, 1, agent_num))

        pool_size1 = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size1, pool_size1))
        print('agent')

    def forward(self, input):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        x1,x2 = input
        B, _, _, _ = x1.shape
        x1 = x1.permute(0, 2, 3, 1).view(B, -1, self.dim)
        x2 = x2.permute(0, 2, 3, 1).view(B, -1, self.dim)
        b, n, c = x1.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads
        qkv1 = self.qkv(x1).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        qkv2 = self.qkv(x2).reshape(b, n, 3, c).permute(2, 0, 1, 3)


        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]  # make torchscript happy (cannot use tensor as tuple)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]


        agent_tokens = self.pool(torch.abs(q1-q2).reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        q1 = q1.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k1 = k1.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v1 = v1.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q2 = q2.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k2 = k2.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v2 = v2.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        #position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
        #position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        #position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        #position_bias = position_bias1 + position_bias2
        

        agent_attn1 = self.softmax((agent_tokens * self.scale) @ k1.transpose(-2, -1) )
        agent_attn2 = self.softmax((agent_tokens * self.scale) @ k2.transpose(-2, -1) )
        agent_attn1 = self.attn_drop(agent_attn1)
        agent_attn2 = self.attn_drop(agent_attn2)
        agent_v1 = agent_attn1 @ v1
        agent_v2 = agent_attn2 @ v2

        #agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
        #agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        #agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        #agent_bias = agent_bias1 + agent_bias2
        #agent_bias = torch.cat([self.ca_bias.repeat(b, 1, 1, 1), agent_bias], dim=-2)
        q_attn1 = self.softmax((q1 * self.scale) @ agent_tokens.transpose(-2, -1) )
        q_attn2 = self.softmax((q2 * self.scale) @ agent_tokens.transpose(-2, -1) )
        q_attn1 = self.attn_drop(q_attn1)
        q_attn2 = self.attn_drop(q_attn2)
        x1 = q_attn1 @ agent_v1
        x2 = q_attn2 @ agent_v2

        x1 = x1.transpose(1, 2).reshape(b, n, c)
        x2 = x2.transpose(1, 2).reshape(b, n, c)

        x1 = x1.view(B, c, int(n**0.5), int(n**0.5))
        x2 = x2.view(B, c, int(n**0.5), int(n**0.5))
        return x1,x2
    
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
    
