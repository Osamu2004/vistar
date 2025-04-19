from model_test.utils.network import get_same_padding,resize
from model_test.nn.act import build_act
from model_test.nn.norm import build_norm
from model_test.utils.list import  val2tuple,list_sum,val2list
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from typing import Tuple
from timm.models.layers import DropPath


class OpSequential(nn.Module):
    def __init__(self, op_list):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x

class SiameseSequential(OpSequential):
    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = input
        for op in self.op_list:
            x1 = op(x1)
            x2 = op(x2)
        return [x1, x2]

class UpSampleLayer(nn.Module):
    def __init__(
        self,
        mode="bicubic",
        size: int or tuple[int, int] or list[int] or None = None,
        factor=2,
        align_corners=False,
    ):
        super(UpSampleLayer, self).__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    @autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.size is not None and tuple(x.shape[-2:]) == self.size) or self.factor == 1:
            return x
        if x.dtype in [torch.float16, torch.bfloat16]:
            x = x.float()
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm="bn2d",
        act_func="relu",
    ):
        super(ConvLayer, self).__init__()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x
    
class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        dropout=0,
        norm=None,
        act_func=None,
    ):
        super(LinearLayer, self).__init__()

        self.dropout = nn.Dropout(dropout, inplace=False) if dropout > 0 else None
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.norm = build_norm(norm, num_features=out_features)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x
    
class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
class LayerScale2d(nn.Module):
    """LayerScale: A learnable scalar to scale the residual branch output.
    """
    def __init__(self, dim, init_value=1e-6):
        super(LayerScale2d, self).__init__()
        self.scale = nn.Parameter(init_value * torch.ones((1, dim, 1, 1)), requires_grad=True)
    def forward(self, x):
        return x * self.scale




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

        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.depth_conv = ConvLayer(
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
    
class TimeMBConv(nn.Module):
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
        super(TimeMBConv, self).__init__()

        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.depth_conv = nn.Conv3d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=(2, 3, 3), padding=(0, 1, 1),groups = mid_channels)
        self.act = build_act(act_func[1])
        
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )

    def forward(self, x1, x2):
        x1 = self.inverted_conv(x1)
        x2 = self.inverted_conv(x2)
        fusion = torch.stack([x1, x2], dim=2)
        fusion = self.act(self.depth_conv(fusion).squeeze(2))
        x1 = fusion +x1
        x2 = fusion +x2
        x1 = self.point_conv(x1)
        x2 = self.point_conv(x2)
        return x1,x2

class ConvolutionalGLU(nn.Module):
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
        act_func=(None, None, None),
    ):
        super(ConvolutionalGLU, self).__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)
        self.fc1 = ConvLayer(
            in_channels,
            mid_channels*2,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.dwconv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
        self.act1 = build_act("gelu")
        self.fc2 = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.dwconv(self.act1(x)) * v
        x = self.fc2(x)
        return x
    
class TFM(nn.Module):
    def __init__(
        self, 
        int_ch: int, 
        use_conv=True, 
        use_bias=True, 
        norm=("ln2d",), 
    ):
        super(TFM, self).__init__()
        if use_conv:
            self.diff_conv = nn.Conv2d(
                    int_ch*2,  
                    int_ch ,
                    kernel_size=1,
                    stride=1,
                    bias=use_bias,
                )
        else:
            self.diff_conv = None
        self.norm = build_norm(norm[0],int_ch )

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2 = inputs
        abs_diff = torch.abs(x1 - x2)
        
        if self.diff_conv is not None:
            return self.norm(self.diff_conv(torch.concat([x1,x2], dim=1))+abs_diff)
        return self.norm(abs_diff)
    
class TFM3d(nn.Module):
    def __init__(
        self, 
        int_ch: int, 
        use_conv=True, 
        use_bias=True, 
        norm=("ln2d",), 
    ):
        super(TFM3d, self).__init__()
        if use_conv:
            self.diff_conv = nn.Conv3d(in_channels=int_ch, out_channels=int_ch, kernel_size=(2, 3, 3), padding=(0, 1, 1),groups = int_ch)
        else:
            self.diff_conv = None
        self.norm = build_norm(norm[0],int_ch )

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2 = inputs
        abs_diff = torch.abs(x1 - x2)
        fusion = torch.stack([x1, x2], dim=2)
        if self.diff_conv is not None:
            return self.norm(self.diff_conv(fusion).squeeze(2)+abs_diff)
        return self.norm(abs_diff)
    


class ResidualBlock(nn.Module): 
    def __init__(
        self,
        main: nn.Module or None,
        shortcut: nn.Module or None,
        post_act=None,
        pre_norm: nn.Module or None = None,
        drop_path_prob: float = 0.0,
        layer_scale: nn.Module or None = None,  
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)
        self.layer_scale = layer_scale

        if drop_path_prob>0.0:
            self.drop_path = DropPath(drop_path_prob)
        else:
            self.drop_path = None

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x)
            if self.layer_scale is not None:
                res = self.layer_scale(res)
            if self.drop_path is not None:
                res = self.drop_path(res)
            res = res + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
                
        return res

class DualResidualBlock(ResidualBlock):
    def forward_main(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pre_norm is None:
            return self.main(x1, x2)
        else:
            return self.main(self.pre_norm(x1),self.pre_norm(x2))
    def forward(self, input:Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x1,x2 = input
        if self.main is None:
            res1 = x1
            res2 = x2
        elif self.shortcut is None:
            res1,res2 = self.forward_main(x1,x2)
        else:
            res1,res2 = self.forward_main(x1,x2)
            if self.layer_scale is not None:
                res1 = self.layer_scale(res1)
                res2 = self.layer_scale(res2)
            if self.drop_path is not None:
                res1 = self.drop_path(res1)
                res2 = self.drop_path(res2)
            res1 = res1 + self.shortcut(x1)
            res2 = res2 + self.shortcut(x2)
            if self.post_act:
                res1 = self.post_act(res1)
                res2 = self.post_act(res2)
                
        return res1,res2
    
class DAGBlock(nn.Module):
    def __init__(
        self,
        inputs,
        merge: str,
        post_input: nn.Module or None,
        middle: nn.Module,
        outputs,
    ):
        super(DAGBlock, self).__init__()

        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        self.merge = merge
        self.post_input = post_input

        self.middle = middle

        self.output_keys = list(outputs.keys())
        self.output_ops = nn.ModuleList(list(outputs.values()))

    def forward(self, feature_dict):
        feat = [op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)]
        if self.merge == "add":
            feat = list_sum(feat)
        elif self.merge == "cat":
            feat = torch.concat(feat, dim=1)
        else:
            raise NotImplementedError
        if self.post_input is not None:
            feat = self.post_input(feat)
        feat = self.middle(feat)
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)
        return feature_dict
    
class SiameseDAGBlock(DAGBlock):
    def __init__(
        self,
        inputs,
        merge: str,
        post_input: nn.Module or None,
        middle: nn.Module,
        outputs,
        fusion: nn.Module  
    ):
        super(SiameseDAGBlock, self).__init__(inputs=inputs, merge=merge, post_input=post_input,middle= middle, outputs=outputs)
        self.fusion = fusion  

    def forward(self, feature_dict_pair):
        feature_dict1, feature_dict2 = feature_dict_pair

        feat1 = [op(feature_dict1[key]) for key, op in zip(self.input_keys, self.input_ops)]
        feat2 = [op(feature_dict2[key]) for key, op in zip(self.input_keys, self.input_ops)]
        if self.merge == "add":
            feat1 = list_sum(feat1)
            feat2 = list_sum(feat2)
        elif self.merge == "cat":
            feat1 = torch.concat(feat1, dim=1)
            feat2 = torch.concat(feat2, dim=1)
        else:
            raise NotImplementedError

        # 可选的后处理操作
        if self.post_input is not None:
            feat1 = self.post_input(feat1)
            feat2 = self.post_input(feat2)

        # 融合层：融合 feat1 和 feat2

        
        feat1 = self.middle(feat1)
        feat2 = self.middle(feat2)
        feat = self.fusion((feat1, feat2))
        feature_dict={}
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)
        return feature_dict
