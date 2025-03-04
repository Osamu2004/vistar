import torch.nn as nn
import torch

from model_test.nn.ops import LinearLayer,ResidualBlock,DualResidualBlock,IdentityLayer,TimeMBConv,LayerScale2d,build_norm,MBConv,ConvLayer
from model_test.nn.act import build_act
from typing import Tuple

from torch.nn import Sequential
import torch.nn.functional as F
class TFM(nn.Module):
    def __init__(
        self, 
        int_ch: int, 
        norm=("ln2d",), 
    ):
        super(TFM, self).__init__()
        self.diff_conv = nn.Conv3d(in_channels=int_ch, out_channels=int_ch, kernel_size=(2, 3, 3), padding=(0, 1, 1),groups = int_ch)
        self.conv11 = nn.Conv2d(in_channels=int_ch*2, out_channels=int_ch, kernel_size=1)
        self.norm = build_norm(norm[0],int_ch )
        self.norm1 = build_norm(norm[0],int_ch )
        self.norm2 = build_norm(norm[0],int_ch )
        self.norm3 = build_norm(norm[0],int_ch )
        self.norm4 = build_norm(norm[0],int_ch )
        self.project = nn.Conv2d(in_channels=int_ch, out_channels=int_ch, kernel_size=1)
        self.act = build_act("gelu")
    
    def forward(self, inputs) -> torch.Tensor:
        x1, x2 = inputs
        fusion = torch.stack([x1, x2], dim=2)
        fusion1 = self.norm1(self.diff_conv(fusion).squeeze(2))
        fusion2 = self.norm2(torch.abs(x1-x2))
        fusion3 = self.norm3(self.conv11(torch.cat([x1,x2],dim=1)))
        
        x = fusion1+fusion2+fusion3+self.norm4(x1)
        x = self.project(x)
        return self.act(self.norm(x))
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
            'b1': [72, 144, 240, 288], 'b2': [96, 192, 320, 384], 'convnext-tiny': [96, 192, 384, 768], 
        }[phi]
        self.fpn_dim = 192
        self.fuse = ConvBnAct(self.fpn_dim*4, self.fpn_dim, 1, 1, 0)
        self.seg = nn.Sequential(ConvBnAct(self.fpn_dim, self.fpn_dim, 1, 1, 0), nn.Conv2d(self.fpn_dim, num_classes*16, 1, 1, 0, bias=True))
        self.out = nn.Conv2d(num_classes*16, num_classes, 3, 1, 1)
        self.ppm = PyramidPoolingModule(self.in_channels[3], self.fpn_dim)
        self.fpn = FeaturePyramidNet(self.fpn_dim)
        self.diff = TFM(self.fpn_dim)


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
        pred = self.out(F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False))
        return pred


import torch.nn as nn
import torch

from model_test.nn.ops import LinearLayer,ResidualBlock,DualResidualBlock,IdentityLayer,TimeMBConv,LayerScale2d,build_norm,MBConv,ConvLayer
from model_test.nn.act import build_act
from typing import Tuple
from model_test.nn.nas import TFM1,TFM2,TFM3,TFM4,TFM5,TFM6,TFM7,TFM8,TFM9,TFM10,NASNetwork

from torch.nn import Sequential
from model_test.nn.nas import TFM1
import torch.nn.functional as F



class Upernet2(nn.Module):
    def __init__(self, num_classes = 1, phi = 'b2'):
        super(Upernet2, self).__init__()
        self.in_channels = {
            'b1': [72, 144, 240, 288], 'b2': [96, 192, 320, 384], 'convnext-tiny': [96, 192, 384, 768],  'convnext-base': [128, 256, 512, 1024],
            'maxvit':[96, 192, 384, 768],'convformer_m36':[96, 192, 384, 576]
        }[phi]
        self.fpn_dim = 256
        self.fuse1 = ConvBnAct(self.fpn_dim*4, self.fpn_dim, 1, 1, 0)
        self.seg2 = nn.Sequential(ConvBnAct(self.fpn_dim, self.fpn_dim, 1, 1, 0), nn.Conv2d(self.fpn_dim, num_classes*16, 1, 1, 0, bias=True))
        self.out2 = nn.Conv2d(num_classes*16, num_classes, 3, 1, 1)
        self.ppm1 = PyramidPoolingModule(self.in_channels[3], self.fpn_dim)
        self.fpn1 = FeaturePyramidNet(self.in_channels,self.fpn_dim)
        self.ppm2 = PyramidPoolingModule(self.in_channels[3], self.fpn_dim)
        self.fpn2 = FeaturePyramidNet(self.in_channels,self.fpn_dim)
        self.diff = TFM(self.fpn_dim)
        self.fuse2 = ConvBnAct(self.fpn_dim*4, self.fpn_dim, 1, 1, 0)
        self.dropout = torch.nn.Dropout2d(p=0.1)
        
        #self.auxiliary_seg = nn.Sequential(ConvBnAct(self.fpn_dim, self.fpn_dim, 1, 1, 0), nn.Conv2d(self.fpn_dim, 2*16, 1, 1, 0, bias=True))
        #self.auxiliary_out = nn.Conv2d(2*16, 2, 3, 1, 1)

    def forward(self, feature_dict_pair):
        feature_dict1, feature_dict2 = feature_dict_pair



        feature_dict1["stage4"] = self.ppm1(feature_dict1["stage4"])
        feature_dict2["stage4"] = self.ppm2(feature_dict2["stage4"])
        fpn1 = self.fpn1(feature_dict1)
        fpn2 = self.fpn2(feature_dict2)
        out_size = fpn1['fpn_layer1'].shape[2:]
        
        list_f1 = []
        list_f1.append(fpn1['fpn_layer1'])
        list_f1.append(F.interpolate(fpn1['fpn_layer2'], out_size, mode='bilinear', align_corners=False))
        list_f1.append(F.interpolate(fpn1['fpn_layer3'], out_size, mode='bilinear', align_corners=False))
        list_f1.append(F.interpolate(fpn1['fpn_layer4'], out_size, mode='bilinear', align_corners=False))

        list_f2 = []
        list_f2.append(fpn2['fpn_layer1'])
        list_f2.append(F.interpolate(fpn2['fpn_layer2'], out_size, mode='bilinear', align_corners=False))
        list_f2.append(F.interpolate(fpn2['fpn_layer3'], out_size, mode='bilinear', align_corners=False))
        list_f2.append(F.interpolate(fpn2['fpn_layer4'], out_size, mode='bilinear', align_corners=False))
        fuse1 = self.fuse1(torch.cat(list_f1, dim=1))
        fuse2 = self.fuse2(torch.cat(list_f2, dim=1))
        diff = self.diff((fuse1,fuse2))

        x = self.seg2(diff)
        pred = self.out2(self.dropout(F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)))

        #auxiliary_x = self.auxiliary_seg(fuse1)
        #auxiliary_pred = self.auxiliary_out(self.dropout(F.interpolate(auxiliary_x, scale_factor=4, mode='bilinear', align_corners=False)))
        output_dict = {
            'main': pred,
        }
        return output_dict
