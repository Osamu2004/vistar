import torch
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torch.nn as nn

from apps.registry import LOSS
# 注册 smp 提供的损失函数
LOSS.register("dice", smp.losses.DiceLoss)
LOSS.register("focal", smp.losses.FocalLoss)


@LOSS.register("kl_div")
class KLDivergenceLoss(nn.Module):
    def __init__(self,):
        """
        初始化 KL 散度损失计算类。
        :param reduction: 损失的归约方法，'mean', 'sum', or 'none'
        """
        super(KLDivergenceLoss, self).__init__()

    
    def forward(self, preds_S, preds_T):
        assert preds_S.shape == preds_T.shape,'the output dim of teacher and student differ'
        N,C,W,H = preds_S.shape
        softmax_pred_T = F.softmax(preds_T.permute(0,2,3,1).contiguous().view(-1,C), dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        loss = (torch.sum( - softmax_pred_T * logsoftmax(preds_S.permute(0,2,3,1).contiguous().view(-1,C))))/W/H
        return loss


@LOSS.register("OHEM")
class OHEMLoss(torch.nn.Module):
    """
    一个自定义损失类，允许传入任意计算损失的函数，而不是指定固定的损失名称。
    
    参数:
        loss_fn (callable): 一个计算损失的函数，接受预测和标签作为输入。
        thresh (float, optional): 阈值，用于选择困难样本，默认是 0.5。
        min_kept (int, optional): 最小保留的困难样本数，默认是 100000。
    """
    
    def __init__(self, loss_fn, thresh=0.7, min_kept=100000,ignore_index=None):
        super().__init__()
        self.loss_fn = loss_fn  # 传入计算损失的函数
        self.thresh = thresh  # 阈值
        self.min_kept = min_kept  # 最小保留困难样本数
        self.ignore_index = ignore_index

    def sample(self, seg_logit, seg_label):
        """选择困难样本并计算样本的权重"""
        batch_kept = self.min_kept * seg_label.size(0)
        # 使用 softmax 计算预测概率
        seg_prob = F.softmax(seg_logit, dim=1)

        # 获取前景类别的预测概率
        tmp_seg_label = seg_label.clone().unsqueeze(1)
        if self.ignore_index is not None:
            tmp_seg_label[tmp_seg_label == self.ignore_index] = 0  # 忽略标签
        seg_prob = seg_prob.gather(1, tmp_seg_label).squeeze(1)

        # 排序并选择最困难的样本
        sort_prob, sort_indices = seg_prob.view(-1).sort()
        # 根据阈值选择困难样本
        min_threshold = sort_prob[min(batch_kept, sort_prob.numel() - 1)]
        threshold = max(min_threshold, self.thresh)
        seg_weight = torch.zeros_like(seg_prob)
        seg_weight[seg_prob < threshold] = 1.0  # 权重为 1 的像素被认为是困难样本
        return seg_weight

    def forward(self, seg_logit, seg_label):
        """计算加权损失"""
        
        # 获取困难样本的权重
        seg_weight = self.sample(seg_logit, seg_label)

        # 使用传入的损失函数计算损失
        loss = self.loss_fn(seg_logit, seg_label)  # 计算损失
        assert loss.shape == seg_weight.shape, f"Shape mismatch: loss shape {loss.shape} and seg_weight shape {seg_weight.shape} do not match."
        weighted_loss = loss * seg_weight  # 应用困难样本的权重
        #print(seg_weight,seg_weight.shape)

        return weighted_loss.mean()  # 返回加权后的损失值