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

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.bn.training:
            raise RuntimeError("BatchNorm should be in evaluation mode (eval) before deployment.")
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
        
        


class RepConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(0, 3, 7, 11), stride=1, dilation=1, groups=1, use_bn=True, deploy=False):
        super(RepConv2d, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = sorted(kernel_sizes)
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.max_kernel_size = max(self.kernel_sizes)
        self.max_padding = (self.max_kernel_size - 1) // 2 * dilation

        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.max_kernel_size, stride=stride,
                                        padding=self.max_padding, dilation=dilation, groups=groups, bias=True)
        else:
            convs = []
            if use_bn:
                if 0 in self.kernel_sizes:
                    assert in_channels == out_channels, "in_channels and out_channels should be equal when kernel_size is 0"
                    assert stride == 1, "stride should be 1 when kernel size is 0"
                    convs.append(nn.BatchNorm2d(out_channels))  # Add BatchNorm for kernel size 0
                    self.kernel_sizes.remove(0)
                # Add ConvBn layers for other kernel sizes
                convs.extend([
                    ConvBn(in_channels, out_channels, k, stride=stride, dilation=dilation,
                           padding=(k - 1) // 2 * dilation, groups=groups)
                    for k in self.kernel_sizes
                ])
            else:
                if 0 in self.kernel_sizes:
                    assert in_channels == out_channels, "in_channels and out_channels should be equal when kernel_size is 0"
                    assert stride == 1, "stride should be 1 when kernel size is 0"
                    convs.append(nn.Identity())  # Identity for kernel size 0
                    self.kernel_sizes.remove(0)
                # Add Conv2d layers for other kernel sizes
                convs.extend([
                    nn.Conv2d(in_channels, out_channels, k, stride=stride, dilation=dilation,
                              padding=(k - 1) // 2 * dilation, bias=True, groups=groups)
                    for k in self.kernel_sizes
                ])
            self.convs = nn.ModuleList(convs)

    def forward(self, x):
        if self.deploy:
            return self.fused_conv(x)
        else:
            conv_outputs = []
            for conv in self.convs:
                conv_outputs.append(conv(x))
            return sum(conv_outputs)

    def _convert_weight_and_bias(self):
        if hasattr(self.convs[-1], 'switch_to_deploy'):
            self.convs[-1].switch_to_deploy()
            weight = self.convs[-1].fused_conv.weight
            bias = self.convs[-1].fused_conv.bias
        else:
            weight = self.convs[-1].weight
            bias = self.convs[-1].bias

        for conv in self.convs[:-1]:
            if isinstance(conv, nn.BatchNorm2d):
                std = (conv.running_var + conv.eps).sqrt()
                t = (conv.weight / std).reshape(-1, 1, 1, 1)
                pad = (self.max_kernel_size - 1) // 2
                input_dim = self.in_channels // self.groups
                identity_weight = F.pad(
                    torch.zeros(self.convs[-1].fused_conv.weight.shape[0], self.convs[-1].fused_conv.weight.shape[1], 1, 1).to(weight.device),
                    [pad, pad, pad, pad]
                )
                for i in range(self.in_channels):
                    identity_weight[i, i % input_dim, self.max_kernel_size//2, self.max_kernel_size//2] = 1
                weight = weight + identity_weight* t
                bias = bias + conv.bias - conv.running_mean * conv.weight / std
            elif isinstance(conv, nn.Identity):
                pad = (self.max_kernel_size - 1) // 2
                input_dim = self.in_channels // self.groups
                identity_weight = F.pad(
                    torch.zeros(self.convs[-1].weight.shape[0], self.convs[-1].weight.shape[1], 1, 1).to(weight.device),
                    [pad, pad, pad, pad]
                ) 
                for i in range(self.in_channels):
                    identity_weight[i, i % input_dim, self.max_kernel_size//2, self.max_kernel_size//2] = 1
                weight = weight + identity_weight
            elif isinstance(conv, ConvBn):
                conv.switch_to_deploy()
                conv_weight = conv.fused_conv.weight
                pad = (self.max_kernel_size - conv.fused_conv.weight.shape[-1]) // 2
                conv_weight = F.pad(conv_weight, [pad, pad, pad, pad])
                conv_bias = conv.fused_conv.bias
                weight = weight + conv_weight
                bias = bias + conv_bias
            elif isinstance(conv, nn.Conv2d):
                conv_weight = conv.weight
                pad = (self.max_kernel_size - conv.weight.shape[-1]) // 2
                conv_weight = F.pad(conv_weight, [pad, pad, pad, pad])
                conv_bias = conv.bias
                weight = weight + conv_weight
                bias = bias + conv_bias
            else:
                raise TypeError(f"Unsupported layer type: {type(conv)}")

        self.weight = weight.detach()
        self.bias = bias.detach()
    @torch.no_grad()
    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        self._convert_weight_and_bias()
        self.fused_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(self.max_kernel_size, self.max_kernel_size),
            stride=self.stride,
            padding=self.max_padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True
        )
        self.fused_conv.weight.data = self.weight
        self.fused_conv.bias.data = self.bias
        del self.convs
        self.deploy = True