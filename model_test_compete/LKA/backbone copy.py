import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model_test_compete.LKA.LKAblock import LKABlock
from model_test.nn.ops import ConvLayer,SiameseSequential
from torch.nn import Sequential

__all__ = [
    "LKABackbone",
    "lka_b1",
    "lka_b2",
    "lka_b3",
]
"需要整理"
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class FeatureEmbedding(nn.Module):
    def __init__(self, num_feature_levels, hidden_dim):
        super(FeatureEmbedding, self).__init__()
        self.num_feature_levels = num_feature_levels
        self.hidden_dim = hidden_dim
        
        # 初始化嵌入层
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
    
    def forward(self, feature1, feature2):
        # 为第一个特征层生成嵌入向量（对应 level 0）
        embed1 = self.level_embed.weight[0][None, :, None,None]
        # 为第二个特征层生成嵌入向量（对应 level 1）
        embed2 = self.level_embed.weight[1][None, :, None,None]
        C = feature1.shape[1]
        embed1_expanded = embed1.view(1, C, 1, 1)
        embed2_expanded = embed2.view(1, C, 1, 1)
        encoded_feature1 = feature1 + embed1_expanded
        encoded_feature2 = feature2 + embed2_expanded
        
        return encoded_feature1, encoded_feature2


from ..base_module import BaseModule
from typing import Dict, Optional, Tuple, Union
class LKABackbone(nn.Module, BaseModel):
    def __init__(
        self,
        width_list,
        depth_list,
        in_channels=3,
        expand_ratio=4,
        img_size=256,
        num_classes=3,
        drop_path_rate = 0.,
        use_layer_scale=True,
    ):
        super().__init__()
        st2_idx = sum(depth_list[:1])
        st3_idx = sum(depth_list[:2])
        st4_idx = sum(depth_list[:3])
        depth = sum(depth_list)
        self.num_classes = num_classes
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]


        self.patch_embed11 = Sequential(
            ConvLayer(
                in_channels=in_channels,
                out_channels=width_list[0],
                stride=1,
                norm="bn2d",
                act_func="gelu",
            ),
            ConvLayer(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=2,
                norm="bn2d",
                act_func=None,
            )
        )
        self.patch_embed12 = Sequential(
            ConvLayer(
                in_channels=1,
                out_channels=width_list[0],
                stride=1,
                norm="bn2d",
                act_func="gelu",
            ),
            ConvLayer(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=2,
                norm="bn2d",
                act_func=None,
            )
        )
        self.num_patches1 = num_patches = img_size // 4
        self.pos_embed1 = nn.Parameter(torch.zeros(1, width_list[0] ,num_patches, num_patches))
        self.FeatureEmbedding1 = FeatureEmbedding(num_feature_levels=2, hidden_dim=width_list[0])
        self.blocks1 = nn.Sequential(*[
            LKABlock(
                dim=width_list[0],  expand_ratio=expand_ratio,
                 drop_path_prob=dpr[i], use_layer_scale=use_layer_scale)
                # use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value, 
                # )
            for i in range(0, st2_idx)])

        self.patch_embed2 = SiameseSequential([ConvLayer(
                in_channels=width_list[0],
                out_channels=width_list[1],
                stride=2,
                norm="bn2d",
                act_func=None,
            )])
        self.num_patches2 = num_patches = num_patches // 2
        self.pos_embed2 = nn.Parameter(torch.zeros(1, width_list[1], num_patches, num_patches))
        self.FeatureEmbedding2 = FeatureEmbedding(num_feature_levels=2, hidden_dim=width_list[1])
        self.blocks2 = nn.Sequential(*[
            LKABlock(
                dim=width_list[1],  expand_ratio=expand_ratio,
                 drop_path_prob=dpr[i],  use_layer_scale=use_layer_scale)
                # use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value, 
                # )
            for i in range(st2_idx,st3_idx)])
        
        self.patch_embed3 = SiameseSequential([ConvLayer(
                in_channels=width_list[1],
                out_channels=width_list[2],
                stride=2,
                norm="bn2d",
                act_func=None,
            )])
        self.num_patches3 = num_patches = num_patches // 2
        self.pos_embed3 = nn.Parameter(torch.zeros(1, width_list[2], num_patches, num_patches))
        self.FeatureEmbedding3 = FeatureEmbedding(num_feature_levels=2, hidden_dim=width_list[2])
        self.blocks3 = nn.Sequential(*[
            LKABlock(
                dim=width_list[2],  expand_ratio=expand_ratio,
                 drop_path_prob=dpr[i],  use_layer_scale=use_layer_scale)
                # use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value, 
                # )
            for i in range(st3_idx, st4_idx)])
        
        self.patch_embed4 = SiameseSequential([ConvLayer(
                in_channels=width_list[2],
                out_channels=width_list[3],
                stride=2,
                norm="bn2d",
                act_func=None,
            )])
        self.num_patches4 = num_patches = num_patches // 2
        self.pos_embed4 = nn.Parameter(torch.zeros(1, width_list[3], num_patches, num_patches))
        self.FeatureEmbedding4 = FeatureEmbedding(num_feature_levels=2, hidden_dim=width_list[3])
        self.blocks4 = nn.Sequential(*[
            LKABlock(
                dim=width_list[3],  expand_ratio=expand_ratio,
                 drop_path_prob=dpr[i], use_layer_scale=use_layer_scale)
                # use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value, 
                # )
            for i in range(st4_idx,depth)])
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        trunc_normal_(self.pos_embed3, std=.02)
        trunc_normal_(self.pos_embed4, std=.02)
        
        self.criterion = nn.CrossEntropyLoss()
        
    def _get_pos_embed(self, pos_embed, num_patches_def, H, W):
        if H * W == num_patches_def * num_patches_def:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed,
                size=(H, W), mode="bilinear")
    def forward_tensor(self, input: torch.Tensor):
        t1_output={}
        t2_output={}
        input[0] = self.patch_embed11(input[0])
        input[1] = self.patch_embed12(input[1])
        B, C, H, W = input[0].shape
        input[0] = input[0] + self._get_pos_embed(self.pos_embed1, self.num_patches1, H, W) 
        input[1] = input[1] + self._get_pos_embed(self.pos_embed1, self.num_patches1, H, W) 
        input = self.blocks1(input)
        t1_output["stage1"]=input[0]
        t2_output["stage1"]=input[1]
           
        input = self.patch_embed2(input)
        B, C, H, W = input[0].shape
        input[0] = input[0] + self._get_pos_embed(self.pos_embed2, self.num_patches2, H, W) 
        input[1] = input[1] + self._get_pos_embed(self.pos_embed2, self.num_patches2, H, W) 
        input[0], input[1] = self.FeatureEmbedding2(input[0], input[1]) 
        input = self.blocks2(input)
        t1_output["stage2"]=input[0]
        t2_output["stage2"]=input[1]

        input = self.patch_embed3(input)
        B, C, H, W = input[0].shape
        input[0] = input[0] + self._get_pos_embed(self.pos_embed3, self.num_patches3, H, W) 
        input[1] = input[1] + self._get_pos_embed(self.pos_embed3, self.num_patches3, H, W) 
        input[0], input[1] = self.FeatureEmbedding3(input[0], input[1]) 
        input = self.blocks3(input)
        t1_output["stage3"]=input[0]
        t2_output["stage3"]=input[1]

        input = self.patch_embed4(input)
        B, C, H, W = input[0].shape
        input[0] = input[0] + self._get_pos_embed(self.pos_embed4, self.num_patches4, H, W) 
        input[1] = input[1] + self._get_pos_embed(self.pos_embed4, self.num_patches4, H, W) 
        input[0], input[1] = self.FeatureEmbedding4(input[0], input[1]) 
        input = self.blocks4(input)
        t1_output["stage4"]=input[0]
        t2_output["stage4"]=input[1]
        return t1_output,t2_output
    def forward(self, batch_inputs: torch.Tensor, data_samples: Union[dict, tuple, list], mode='tensor'):
        data_samples = torch.stack(data_samples)
        if mode == 'tensor':
            return self.forward_tensor(batch_inputs)
        elif mode == 'predict':
            feats = self.forward_tensor(batch_inputs)
            predictions = torch.argmax(feats, 1)
            return feats, predictions
        elif mode == 'loss':
            feats = self.forward_tensor(batch_inputs)
            loss = self.criterion(feats, data_samples)
            return feats, dict(loss=loss)


def lka_b0():
    backbone = LKABackbone(
            width_list=[72, 144, 240, 288], depth_list=[1, 1, 1, 1], 
             drop_path_rate=0.2,use_layer_scale=True)
    return backbone


def lka_b1():
    backbone = LKABackbone(
            width_list=[36, 72, 120, 144], depth_list=[1, 1, 3, 1], 
             drop_path_rate=0.2,use_layer_scale=True)
    return backbone
        
def lka_b2():
    backbone = LKABackbone(
            width_list=[96, 192, 320, 384], depth_list=[1, 1, 9, 1], 
             drop_path_rate=0.4,use_layer_scale=True)
    return backbone

def lka_b3():
    backbone = LKABackbone(
            width_list=[96, 192, 384, 512], depth_list=[4, 6, 14, 6], 
             drop_path_rate=0.5,use_layer_scale=True)
    return backbone