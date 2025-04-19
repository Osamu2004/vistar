import torch
import torch.nn as nn
import timm
import math
from timm.models.convnext import _cfg
from model_test.nn.act import build_act
from torch.nn import Sequential
from model_test.nn.ops import LinearLayer,build_norm





def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    """
    if name.startswith('backbone'):
        return
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
    elif isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
            
import torch.nn as nn
import torch

from model_test.nn.ops import LinearLayer,ResidualBlock,DualResidualBlock,IdentityLayer,TimeMBConv,LayerScale2d,build_norm,MBConv,ConvLayer
from model_test.nn.act import build_act
from typing import Tuple

from torch.nn import Sequential
import torch.nn.functional as F

class ConvBnAct(nn.Module):

    def __init__(self, in_channel, out_channel, kernel, stride, padding, dilation=1, bias=False, act=True):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = build_norm("ln2d",num_features=out_channel)
        self.act = build_act("gelu") if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class PyramidPoolingModule(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(PyramidPoolingModule, self).__init__()
        inter_channels = in_channels //4
        self.cba1 = ConvBnAct(in_channels, inter_channels, 1, 1, 0)
        self.cba2 = ConvBnAct(in_channels, inter_channels, 1, 1, 0)
        self.cba3 = ConvBnAct(in_channels, inter_channels, 1, 1, 0)
        self.cba4 = ConvBnAct(in_channels, inter_channels, 1, 1, 0)
        self.out  = ConvBnAct(in_channels * 2, out_channels, 1, 1, 0)

    def pool(self, x, size):
        return nn.AdaptiveAvgPool2d(size)(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode="bilinear", align_corners=True)

    def forward(self, x):
        size = x.shape[2:]
        f1 = self.upsample(self.cba1(self.pool(x, 1)), size)
        f2 = self.upsample(self.cba2(self.pool(x, 2)), size)
        f3 = self.upsample(self.cba3(self.pool(x, 3)), size)
        f4 = self.upsample(self.cba4(self.pool(x, 6)), size)
        f = torch.cat([x, f1, f2, f3, f4], dim=1)
        return self.out(f)

class FeaturePyramidNet(nn.Module):

    def __init__(self,list, fpn_dim=256):
        self.fpn_dim = fpn_dim
        super(FeaturePyramidNet, self).__init__()
        self.fpn_in = nn.ModuleDict({'fpn_layer1': ConvBnAct(list[0] , self.fpn_dim, 1, 1, 0), 
                                     "fpn_layer2": ConvBnAct(list[1], self.fpn_dim, 1, 1, 0), 
                                     "fpn_layer3": ConvBnAct(list[2], self.fpn_dim, 1, 1, 0), 
                                    })
        self.fpn_out = nn.ModuleDict({'fpn_layer1': ConvBnAct(self.fpn_dim, self.fpn_dim, 3, 1, 1), 
                                      "fpn_layer2": ConvBnAct(self.fpn_dim, self.fpn_dim, 3, 1, 1), 
                                      "fpn_layer3": ConvBnAct(self.fpn_dim, self.fpn_dim, 3, 1, 1), 
                                      })

    def forward(self, pyramid_features):
        """
        
        """
        fpn_out = {}
        
        f = pyramid_features["stage4"]
        fpn_out['fpn_layer4'] = f
        x = self.fpn_in['fpn_layer3'](pyramid_features["stage3"])
        f = F.interpolate(f, x.shape[2:], mode='bilinear', align_corners=False)
        f = x + f
        fpn_out['fpn_layer3'] = self.fpn_out['fpn_layer3'](f)

        x = self.fpn_in['fpn_layer2'](pyramid_features["stage2"])
        f = F.interpolate(f, x.shape[2:], mode='bilinear', align_corners=False)
        f = x + f
        fpn_out['fpn_layer2'] = self.fpn_out['fpn_layer2'](f)

        x = self.fpn_in['fpn_layer1'](pyramid_features["stage1"])
        f = F.interpolate(f, x.shape[2:], mode='bilinear', align_corners=False)
        f = x + f
        fpn_out['fpn_layer1'] = self.fpn_out['fpn_layer1'](f)

        return fpn_out


class Upernet(nn.Module):
    def __init__(self, num_classes = 1, phi = 'b2'):
        super(Upernet, self).__init__()
        self.in_channels = {
            'b1': [72, 144, 240, 288], 'b2': [96, 192, 320, 384], 'convnext-small': [96, 192, 384, 768], 'efficientnetv2':[48,80,176,512],
            'convnext-base': [128, 256, 512, 1024],'mambaout_small':[96,192,384,576]
        }[phi]
        self.fpn_dim = 256
        self.fuse = ConvBnAct(self.fpn_dim*4, self.fpn_dim, 1, 1, 0)
        self.seg = nn.Sequential(ConvBnAct(self.fpn_dim, self.fpn_dim, 1, 1, 0), nn.Conv2d(self.fpn_dim, num_classes*16, 1, 1, 0, bias=True))
        self.out = nn.Conv2d(num_classes*16, num_classes, 3, 1, 1)
        self.ppm = PyramidPoolingModule(self.in_channels[3], self.fpn_dim)
        self.fpn = FeaturePyramidNet(self.in_channels,self.fpn_dim)
        self.dropout = torch.nn.Dropout2d(p=0.1)

    def forward(self, feature_dict):

        feature_dict["stage4"] = self.ppm(feature_dict["stage4"])
        fpn1 = self.fpn(feature_dict)
        out_size = fpn1['fpn_layer1'].shape[2:]
        
        list_f1 = []
        list_f1.append(fpn1['fpn_layer1'])
        list_f1.append(F.interpolate(fpn1['fpn_layer2'], out_size, mode='bilinear', align_corners=False))
        list_f1.append(F.interpolate(fpn1['fpn_layer3'], out_size, mode='bilinear', align_corners=False))
        list_f1.append(F.interpolate(fpn1['fpn_layer4'], out_size, mode='bilinear', align_corners=False))

        fuse1 = self.fuse(torch.cat(list_f1, dim=1))

        x = self.seg(fuse1)
        pred = self.out(self.dropout(F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)))
        return pred
    
class segmentor_convnext(nn.Module):
    def __init__(self, dataset=''):
        super().__init__()
        config1 = _cfg(url='', file='/opt/data/private/competition/code/model_hky/convnext_small_fb_in22k.safetensors')
        self.backbone1 = timm.create_model('convnext_small.fb_in22k',features_only=True,pretrained=True,pretrained_cfg=config1,drop_path_rate=0.4)
        self.phi = 'convnext-small'
        self.decoder_pretrain = Upernet(num_classes=2, phi=self.phi)
    def init_weights(self, mode=''):
        self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)
    def forward(self, x):
        t1 = x["image"]
        feed_dict1 = self.backbone1(t1)

        dict1 = {f"stage{i+1}": feat for i, feat in enumerate(feed_dict1)}
        output = self.decoder_pretrain(dict1)
        return output
    
class segmentor_mambaout(nn.Module):
    def __init__(self, dataset=''):
        super().__init__()
        config1 = _cfg(url='', file='/opt/data/private/competition/code/model_pretrain/mambaout_small.in1k.safetensors')
        self.backbone1 = timm.create_model('mambaout_small.in1k',features_only=True,pretrained=True,pretrained_cfg=config1,drop_path_rate=0.4)
        self.phi = 'mambaout_small'
        self.decoder_pretrain = Upernet(num_classes=2, phi=self.phi)
    def init_weights(self, mode=''):
        self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)
    def forward(self, x):
        t1 = x["image"]
        feed_dict1 = self.backbone1(t1)
        feed_dict1 = [x.permute(0, 3, 1, 2) for x in feed_dict1]
        
        dict1 = {f"stage{i+1}": feat for i, feat in enumerate(feed_dict1) }
        output = self.decoder_pretrain(dict1)
        return output
    
class segmentor_inception_next(nn.Module):
    def __init__(self, dataset=''):
        super().__init__()
        config1 = _cfg(url='', file='/opt/data/private/competition/code/model_pretrain/inception_next_small.sail_in1k.safetensors')
        self.backbone1 = timm.create_model('inception_next_small.sail_in1k',features_only=True,pretrained=True,pretrained_cfg=config1,drop_path_rate=0.4)
        self.phi = 'convnext-small'
        self.decoder_pretrain = Upernet(num_classes=2, phi=self.phi)
    def init_weights(self, mode=''):
        self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)
    def forward(self, x):
        t1 = x["image"]
        feed_dict1 = self.backbone1(t1)

        dict1 = {f"stage{i+1}": feat for i, feat in enumerate(feed_dict1)}

        output = self.decoder_pretrain(dict1)
        return output