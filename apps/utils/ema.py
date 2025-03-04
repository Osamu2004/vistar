# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import copy
import math

import torch
import torch.nn as nn

from apps.utils.model import is_parallel

__all__ = ["EMA"]


def update_ema(ema: nn.Module, new_state_dict: dict[str, torch.Tensor], decay: float) -> None:
    for k, v in ema.state_dict().items():
        if v.dtype.is_floating_point:
            v -= (1.0 - decay) * (v - new_state_dict[k].detach())


class EMA:
    def __init__(self, model: nn.Module, decay: float, warmup_steps=2000):
        self.shadows = copy.deepcopy(model.module if is_parallel(model) else model).eval()
        self.decay = decay
        self.warmup_steps = warmup_steps

        for p in self.shadows.parameters():
            p.requires_grad = False

    def step(self, model: nn.Module, global_step: int) -> None:
        with torch.no_grad():
            msd = (model.module if is_parallel(model) else model).state_dict()
            update_ema(self.shadows, msd, self.decay * (1 - math.exp(-global_step / self.warmup_steps)))

    def state_dict(self) -> dict[float, dict[str, torch.Tensor]]:
        return {self.decay: self.shadows.state_dict()}

    def load_state_dict(self, state_dict: dict[float, dict[str, torch.Tensor]]) -> None:
        for decay in state_dict:
            if decay == self.decay:

                
                # 加载更新后的 state_dict
                self.shadows.load_state_dict(state_dict[decay])

class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """
    def __init__(self, model, decay, device='cpu'):
        ema_avg = (lambda avg_model_param, model_param, num_averaged:
                   decay * avg_model_param + (1 - decay) * model_param)
        super().__init__(model, device, ema_avg)

    def update_parameters(self, model):
        for p_swa, p_model in zip(self.module.state_dict().values(), model.state_dict().values()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model_,
                                     self.n_averaged.to(device)))
            self.n_averaged += 1
