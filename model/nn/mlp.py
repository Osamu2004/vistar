from .build_act import build_act
from .build_norm import build_norm
from .conv2d import ConvNormAct
import torch
import torch.nn as nn
from .utils import val2tuple
class Mlp2d(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expand_ratio=4,
            mid_channels=None,
            use_bias=(False,False),
            drop_ratio=(0., 0.),
            norm=(None, None),
            act_func=("relu6", None),
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        mid_channels = mid_channels or round(in_channels * expand_ratio)
        use_bias = use_bias 

        self.fc1 = nn.Conv2d(in_channels, mid_channels, 1, bias=use_bias[0])
        self.act = build_act(act_func[0])
        self.drop1 = nn.Dropout(drop_ratio[0])
        self.norm = build_norm(norm[0], num_features=mid_channels) if norm[0] else nn.Identity()
        self.fc2 = nn.Conv2d(mid_channels, out_channels, bias=use_bias[1])
        self.drop2 = nn.Dropout(drop_ratio[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MBConv(nn.Module):
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
        super(MBConv, self).__init__()

        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = ConvNormAct(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.depth_conv = ConvNormAct(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x
    

