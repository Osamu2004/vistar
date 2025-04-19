import torch
import torch.nn as nn
import torch.nn.functional as F
from model_test.nn.norm import build_norm
from model_test.nn.act import build_act
from typing import Tuple

class TFM1(nn.Module):
    def __init__(
        self, 
        int_ch: int, 
        norm=("ln2d",), 
    ):
        super(TFM1, self).__init__()
        self.diff_conv = nn.Conv3d(in_channels=int_ch, out_channels=int_ch, kernel_size=(2, 3, 3), padding=(0, 1, 1),groups = int_ch)
        self.conv11 = nn.Conv2d(in_channels=int_ch*2, out_channels=int_ch, kernel_size=1)
        self.norm = build_norm(norm[0],int_ch )
        self.norm1 = build_norm(norm[0],int_ch )
        self.norm2 = build_norm(norm[0],int_ch )
        self.norm3 = build_norm(norm[0],int_ch )
        self.mid_gelu1 = build_act("gelu")
        self.mid_gelu2 = build_act("gelu")
        self.project = nn.Conv2d(in_channels=int_ch*3, out_channels=int_ch, kernel_size=1)

    def channel(self, F1, F2):

        """
        Compute x3[:, i, :, :] = sum_j (F1[:, i, :, :] * F2[:, j, :, :])

        Args:
            F1: Tensor of shape (B, C1, H, W)
            F2: Tensor of shape (B, C2, H, W)

        Returns:
            x3: Tensor of shape (B, C1, H, W)
        """
        # Use efficient einsum for the operation
        x3 = torch.einsum('bihw,bjhw->bihw', F1, F2)

        return x3
    
    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2 = inputs
        fusion = torch.stack([x1, x2], dim=2)
        fusion1 = self.mid_gelu1(self.norm1(self.diff_conv(fusion).squeeze(2)))
        fusion2 = self.norm2(torch.abs(x1-x2))
        fusion3 = self.mid_gelu2(self.norm3(self.conv11(torch.cat([x1,x2],dim=1))))
        x = torch.cat((fusion1, fusion2,fusion3), dim=1)
        x = self.project(x)
        return self.norm(x)

class TFM2(nn.Module):
    def __init__(
        self, 
        int_ch: int, 
        norm=("ln2d",), 
    ):
        super(TFM2, self).__init__()
        self.diff_conv = nn.Conv2d(in_channels=int_ch*2, out_channels=int_ch, kernel_size=1)
        self.norm = build_norm(norm[0],int_ch )

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2 = inputs
        return self.norm(self.diff_conv(torch.concat([x1,x2], dim=1)))
    
class TFM3(nn.Module):
    def __init__(
        self, 
        int_ch: int, 
        norm=("ln2d",), 
    ):
        super(TFM3, self).__init__()
        self.norm = build_norm(norm[0],int_ch )

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2 = inputs
        return self.norm(x1+x2)
    
class TFM4(nn.Module):
    def __init__(
        self, 
        int_ch: int, 
        norm=("ln2d",), 
    ):
        super(TFM4, self).__init__()
        self.norm = build_norm(norm[0],int_ch )

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2 = inputs
        return self.norm(torch.abs(x1-x2))

class TFM5(nn.Module):
    def __init__(
        self, 
        int_ch: int, 
        norm=("bn2d",), 
    ):
        super(TFM5, self).__init__()
        self.diff_conv = nn.Conv3d(in_channels=int_ch, out_channels=int_ch, kernel_size=(2, 3, 3), padding=(0, 1, 1),groups = int_ch)
        self.norm = build_norm(norm[0],int_ch )

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2 = inputs
        fusion = torch.stack([x1, x2], dim=2)
        return self.norm(self.diff_conv(fusion).squeeze(2))
    
class TFM6(nn.Module):
    def __init__(self, int_ch: int, norm=("ln2d",)):
        super(TFM6, self).__init__()
        self.weights = nn.Parameter(torch.ones(1, int_ch, 1, 1))  # 可学习的权重
        self.norm = build_norm(norm[0], int_ch)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2 = inputs
        fusion = self.weights * x1 * x2
        return self.norm(fusion)


class TFM7(nn.Module):
    def __init__(
        self, 
        int_ch: int, 
    ):
        super(TFM7, self).__init__()
        self.diff_conv = nn.Conv3d(in_channels=int_ch, out_channels=int_ch, kernel_size=(2, 3, 3), padding=(0, 1, 1),groups = int_ch)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2 = inputs
        fusion = torch.stack([x1, x2], dim=2)
        return self.diff_conv(fusion).squeeze(2)


class TFM8(nn.Module):
    def __init__(self, int_ch: int, norm=("ln2d",)):
        super(TFM8, self).__init__()
        self.query_conv = nn.Conv2d(int_ch, int_ch, kernel_size=1)
        self.key_conv = nn.Conv2d(int_ch, int_ch, kernel_size=1)
        self.value_conv = nn.Conv2d(int_ch, int_ch, kernel_size=1)
        self.norm = build_norm(norm[0], int_ch)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2 = inputs
        # Compute attention
        query = self.query_conv(x1)  # (B, C, H, W)
        key = self.key_conv(x2)
        value = self.value_conv(x2)
        attention = torch.softmax(torch.einsum('bchw,bcij->bhwij', query, key), dim=-1)
        fusion = torch.einsum('bhwij,bcij->bchw', attention, value)
        return self.norm(fusion)
    
class TFM9(nn.Module):
    def __init__(self, int_ch: int, reduction: int = 16, norm=("ln2d",)):
        super(TFM9, self).__init__()
        # 通道注意力分支
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc1 = nn.Conv2d(int_ch*2, int_ch // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(int_ch // reduction, int_ch * 2, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)  # 用于通道权重归一化
        # 最终融合
        self.norm = build_norm(norm[0], int_ch)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2 = inputs
        # 特征叠加
        combined = torch.stack([x1, x2], dim=1)  # (B, 2, C, H, W)
        B, N, C, H, W = combined.size()

        # 计算通道注意力
        combined_flat = combined.view(B, N * C, H, W)  # 展平通道维度
        pooled = self.global_pool(combined_flat)  # (B, N*C, 1, 1)
        attention = self.fc2(self.relu(self.fc1(pooled)))  # (B, N*C, 1, 1)
        attention = self.softmax(attention.view(B, 2, C, 1, 1))  # (B, 2, C, 1, 1)

        # 选择性加权融合
        x1_weighted = attention[:, 0] * x1
        x2_weighted = attention[:, 1] * x2
        fusion = x1_weighted + x2_weighted  # 融合操作

        return self.norm(fusion)
    
class TFM10(nn.Module):
    def __init__(self, int_ch: int, norm=("ln2d",)):
        super(TFM10, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(in_channels=int_ch * 2, out_channels=int_ch // 4, kernel_size=1, bias=False),  # 降维
            nn.ReLU(),
            nn.Conv2d(in_channels=int_ch // 4, out_channels=int_ch * 2, kernel_size=1, bias=False),  # 升维
            nn.Sigmoid()  # 输出权重
        )
        self.diff_conv = nn.Conv2d(in_channels=int_ch * 2, out_channels=int_ch, kernel_size=1)  # 融合后特征压缩
        self.norm = build_norm(norm[0], int_ch)  # 归一化

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2 = inputs
        # 拼接特征
        combined = torch.cat([x1, x2], dim=1)  # Shape: (B, 2*C, H, W)

        # SE 注意力生成
        attention = self.se(combined)  # Shape: (B, 2*C, 1, 1)

        # 加权特征
        weighted = combined * attention  # 按通道加权

        # 特征融合
        fused = self.diff_conv(weighted)  # 压缩到原始通道数
        return self.norm(fused)
    

    
class NASNetwork(nn.Module):
    def __init__(self, input_channels, modules, selection_method="softmax"):
        super(NASNetwork, self).__init__()
        #assert isinstance(modules, list), "Modules must be a list of nn.Module instances"
        self.module_list = nn.ModuleList(modules)  # 使用 nn.ModuleList 封装
        self.num_modules = len(modules)
        self.selection_method = selection_method

        # 定义权重生成器
        self.weight_generator = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_channels, self.num_modules)
        )

    def forward(self, input):
        # 合并输入
        x1, x2 = input
        combined_input = torch.cat([x1, x2], dim=1)
        module_scores = self.weight_generator(combined_input)

        if self.selection_method == "softmax":
            module_weights = F.softmax(module_scores, dim=1)
            module_outputs = [module((x1, x2)) for module in self.module_list]
            module_outputs = torch.stack(module_outputs, dim=1)
            module_weights = module_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            output = (module_outputs * module_weights).sum(dim=1)

        elif self.selection_method == "topk":
            module_index = torch.argmax(module_scores, dim=1)
            output = torch.stack([self.modules[idx]((x1[i].unsqueeze(0), x2[i].unsqueeze(0)))
                                  for i, idx in enumerate(module_index)])
        else:
            raise ValueError("Invalid selection_method. Choose 'softmax' or 'topk'.")

        return output



