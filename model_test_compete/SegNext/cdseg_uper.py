import torch
import torch.nn as nn
import math
from torch.nn.modules.batchnorm import _BatchNorm
from model_test.nn.ops import (
    ConvLayer,
    SiameseDAGBlock,
    IdentityLayer,
    MBConv,
    ResidualBlock,
    UpSampleLayer,
    OpSequential
)
from model_test.nn.ops import DAGBlock
from model_test.nn.nas import TFM1,TFM2,TFM3,TFM4,TFM5,TFM6,TFM7,TFM8,TFM9,TFM10,NASNetwork

__all__ = [

    "segnext_seg_uperb1",
]
from model_test.nn.ops import LinearLayer,ResidualBlock,DualResidualBlock,IdentityLayer,TimeMBConv,LayerScale2d,build_norm,MBConv,ConvLayer
from model_test.nn.act import build_act
from torch.nn import Sequential


from model_test_compete.LKA.upernet import Upernet

class CBFCDSeg(nn.Module):
    def __init__(self, backbone, head: Upernet) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t1,t2 = x["image"],x["t2_image"]
        feed_dict = self.backbone([t1,t2])
        feed_dict = self.head(feed_dict)

        return feed_dict
    
def segnext_seg_uperb1(dataset: str) -> CBFCDSeg:
    from model_test_compete.SegNext.backbone import segnext_b1

    backbone = segnext_b1()
    if dataset == "levir_cd":
        head = Upernet(
            fid_list=["stage4", "stage3", "stage2","stage1"],
            in_channel_list=[288, 240, 144,72],
            stride_list=[32, 16, 8,4],
            head_stride=4,
            head_width=144,
            head_depth=2,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=4,
            num_classes=2,
        )
    elif dataset == "dfc25_track2":
        head = Upernet(
             phi = 'b1',

            num_classes=4,
        )
    else:
        raise NotImplementedError
    model = CBFCDSeg(backbone, head)
    return model

def lka_seg_lkauperb2(dataset: str) -> CBFCDSeg:
    from model_test_compete.LKA.backbone import lka_b2
    backbone = lka_b2()

    if dataset == "levir_cd":
        head = Upernet(
            fid_list=["stage4", "stage3", "stage2","stage1"],
            in_channel_list=[384, 320, 192, 96],
            stride_list=[32, 16, 8, 4],
            head_stride=4,
            head_width=192,
            head_depth=3,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=4,
            n_classes=2,
        )
    elif dataset == "dfc25_track2":
        head = Upernet(
             phi = 'b2',

            num_classes=4,
        )
    else:
        raise NotImplementedError
    model = CBFCDSeg(backbone, head)
    return model
