import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from .base_module import BaseModule,Sequential
from .backbone import Backbone
from apps.registry import MODEL
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d']




def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(BaseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg=init_cfg)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,init_cfg=None):
        super(Bottleneck, self).__init__(init_cfg=init_cfg)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(Backbone):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, deep_stem=False,pretrained=None, init_cfg=None):
        super(ResNet, self).__init__(init_cfg=init_cfg)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.zero_init_residual = zero_init_residual
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if pretrained:
            self.init_cfg = pretrained
            block_init_cfg = None
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer='BatchNorm2d')
                ]
                # Zero-initialize the last BN in each residual branch,
                # so that the residual branch starts with zeros, and each residual block behaves like an identity.
                # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='bn2'))
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='bn3'))
                else:
                    block_init_cfg = None

        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.deep_stem = deep_stem
        if deep_stem:
            self.stem = Sequential(
                nn.Conv2d(3, self.inplanes // 2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(self.inplanes // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inplanes // 2, self.inplanes // 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.inplanes // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inplanes // 2, self.inplanes, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],init_cfg=block_init_cfg)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],init_cfg=block_init_cfg)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],init_cfg=block_init_cfg)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],init_cfg=block_init_cfg)





    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,init_cfg=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,init_cfg=init_cfg))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,init_cfg=init_cfg))

        return Sequential(*layers)

    def stem_forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        return x

    def forward(self, x):
        x = self.stem_forward(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        return x

from .weight_url import weight_urls
def _resnet( arch,block, layers, pretrained, **kwargs) -> nn.Module:
    if pretrained:
        url = weight_urls.get(arch)
        if not url:
            raise ValueError(f"Model '{arch}' not found in weight URLs")
        pretrained_config = dict(type='Pretrained', checkpoint=url,url=True)
    model = ResNet(block, layers, pretrained = pretrained_config,**kwargs)

    return model


def resnet18(pretrained, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNet( BasicBlock, [2, 2, 2, 2], pretrained=pretrained,
                   **kwargs)


def resnet34(pretrained, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNet( BasicBlock, [3, 4, 6, 3], pretrained=pretrained, 
                   **kwargs)


def resnet50(pretrained, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet( Bottleneck, [3, 4, 6, 3], pretrained=pretrained, 
                   **kwargs)


def resnet101(pretrained, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet( Bottleneck, [3, 4, 23, 3], pretrained=pretrained, 
                   **kwargs)


def resnet152(pretrained, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(Bottleneck, [3, 8, 36, 3], pretrained=pretrained, 
                   **kwargs)


def resnext50_32x4d(pretrained, **kwargs):
    """Constructs a ResNeXt-50 32x4d model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet( Bottleneck, [3, 4, 6, 3], pretrained=pretrained,
                     **kwargs)


def resnext101_32x4d(pretrained, **kwargs):
    """Constructs a ResNeXt-101 32x4d model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet( Bottleneck, [3, 4, 23, 3], pretrained=pretrained,
                    **kwargs)


def resnext101_32x8d(pretrained, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet( Bottleneck, [3, 4, 23, 3],
                   pretrained=pretrained,  **kwargs)


def resnet50_v1c(pretrained, **kwargs) -> nn.Module:
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet( Bottleneck, [3, 4, 6, 3], pretrained=pretrained, deep_stem=True,
                   **kwargs)


def resnet101_v1c(pretrained, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet( Bottleneck, [3, 4, 23, 3], pretrained=pretrained,  deep_stem=True,
                   **kwargs)


MODEL.register('resnet18', resnet18)
MODEL.register('resnet34', resnet34)
MODEL.register('resnet50', resnet50)
MODEL.register('resnet101', resnet101)
MODEL.register('resnext50_32x4d', resnext50_32x4d)
MODEL.register('resnext101_32x4d', resnext101_32x4d)
MODEL.register('resnext101_32x8d', resnext101_32x8d)
MODEL.register('resnet50_v1c', resnet50_v1c)
MODEL.register('resnet101_v1c', resnet101_v1c)