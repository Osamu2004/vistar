import torch
import torch.nn as nn
from model.nn.utils import get_same_padding
from model.nn.build_norm import build_norm
from model.nn.build_act import build_act

class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias="auto",
        dropout=0,
        norm="bn2d",
        act_func="relu",
    ):
        super(ConvNormAct, self).__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)
        self.with_norm = norm is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x
    
class ConvBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(ConvBn, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.bn = nn.BatchNorm2d(num_features=out_channels)


    def _fuse_bn_tensor(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std


    def switch_to_deploy(self):
        deploy_k, deploy_b = self._fuse_bn_tensor(self.conv, self.bn)
        self.deploy = True
        self.fused_conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels,
                                    kernel_size=self.conv.kernel_size, stride=self.conv.stride,
                                    padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups, bias=True,
                                    padding_mode=self.conv.padding_mode)
        self.__delattr__('conv')
        self.__delattr__('bn')
        self.fused_conv.weight.data = deploy_k
        self.fused_conv.bias.data = deploy_b

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.conv(input)
            square_outputs = self.bn(square_outputs)
            return square_outputs