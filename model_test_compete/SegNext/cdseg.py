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
    "lka_seg_lkab1",
    "lka_seg_lkab2",
    "lka_seg_lkab3",
]
from model_test.nn.ops import LinearLayer,ResidualBlock,DualResidualBlock,IdentityLayer,TimeMBConv,LayerScale2d,build_norm,MBConv,ConvLayer
from model_test.nn.act import build_act
from torch.nn import Sequential


class SegHead(SiameseDAGBlock):
    def __init__(
        self,
        fid_list,
        in_channel_list,
        stride_list,
        head_stride: int,
        head_width: int,
        head_depth: int,
        expand_ratio: float,
        middle_op: str,
        final_expand: float or None,
        n_classes: int,
        dropout=0,
        norm="ln2d",
        act_func="gelu",
    ):
        inputs = {}
        for fid, in_channel, stride in zip(fid_list, in_channel_list, stride_list):
            factor = stride // head_stride
            if factor == 1:
                inputs[fid] = ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None)
            else:
                inputs[fid] = OpSequential(
                    [
                        ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None),
                        UpSampleLayer(factor=factor),
                    ]
                )

        middle = []
        for _ in range(head_depth):
            if middle_op == "mbconv":
                block = MBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=(norm, norm, norm),
                    act_func=(act_func, act_func, None),
                )
            else:
                raise NotImplementedError
            middle.append(ResidualBlock(block, IdentityLayer()))
        middle = OpSequential(middle)

        outputs = {
            "segout": OpSequential(
                [
                    (
                        None
                        if final_expand is None
                        else ConvLayer(head_width, n_classes*16, 1, norm=norm, act_func=act_func)
                    ),
                    UpSampleLayer(factor=head_stride),
                    ConvLayer(
                        n_classes*16,
                        n_classes,
                        3,
                        use_bias=True,
                        dropout=dropout,
                        norm=None,
                        act_func=None,
                    ),
                ]
            )
        }
        fusion = TFM1(head_width)
        post_input = ConvLayer(
                    head_width*4,
                    head_width,
                    1,
                    use_bias=True,
                    norm=norm,
                    act_func=act_func
                )

        super(SegHead, self).__init__(inputs, "cat", middle=middle, outputs=outputs ,fusion=fusion,post_input=post_input,)

class CBFCDSeg(nn.Module):
    def __init__(self, backbone, head: SegHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t1,t2 = x["image"],x["t2_image"]
        feed_dict = self.backbone([t1,t2])
        feed_dict = self.head(feed_dict)

        return feed_dict["segout"]

def lka_seg_lkab0(dataset: str) -> CBFCDSeg:
    from model_test_compete.LKA.backbone import lka_b0

    backbone = lka_b0()
    if dataset == "levir_cd":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2","stage1"],
            in_channel_list=[288, 240, 144,72],
            stride_list=[32, 16, 8,4],
            head_stride=4,
            head_width=144,
            head_depth=2,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=4,
            n_classes=2,
        )
    elif dataset == "dfc25_track2":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2","stage1"],
            in_channel_list=[288, 240, 144,72],
            stride_list=[32, 16, 8,4],
            head_stride=4,
            head_width=144,
            head_depth=2,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=4,
            n_classes=4,
        )
    else:
        raise NotImplementedError
    model = CBFCDSeg(backbone, head)
    return model





def lka_seg_segnextb1(dataset: str) -> CBFCDSeg:
    from model_test_compete.SegNext.backbone import segnext_b1

    backbone = segnext_b1()
    if dataset == "levir_cd":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2","stage1"],
            in_channel_list=[288, 240, 144,72],
            stride_list=[32, 16, 8,4],
            head_stride=4,
            head_width=144,
            head_depth=2,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=4,
            n_classes=2,
        )
    elif dataset == "dfc25_track2":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2","stage1"],
            in_channel_list=[288, 240, 144,72],
            stride_list=[32, 16, 8,4],
            head_stride=4,
            head_width=144,
            head_depth=2,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=4,
            n_classes=4,
        )
    else:
        raise NotImplementedError
    model = CBFCDSeg(backbone, head)
    return model

def cbf_seg_testb2(dataset: str) -> CBFCDSeg:
    from model_test_compete.SegNext.backbone import segnext_b2
    backbone = segnext_b2()

    if dataset == "levir_cd":
        head = SegHead(
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
    elif dataset == "s2looking":
        head = SegHead(
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
    else:
        raise NotImplementedError
    model = CBFCDSeg(backbone, head)
    return model
