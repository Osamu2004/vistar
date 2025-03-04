# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import json

import numpy as np
import torch.nn as nn

from apps.builder import make_optimizer,make_loss
from apps.utils.lr import CosineLRwithWarmup

__all__ = [ "RunStepConfig"]


class Scheduler:
    PROGRESS = 0

class RunStepConfig:
    n_steps: int  # 总训练步数
    init_lr: float
    warmup_steps: int  # warmup 阶段步数
    warmup_lr: float
    lr_schedule_name: str
    lr_schedule_param: dict
    optimizer_name: str
    optimizer_params: dict
    weight_decay: float
    no_wd_keys: list
    grad_clip: float  # allow None to turn off grad clipping
    loss: dict
    @property
    def none_allowed(self):
        return ["grad_clip", "eval_image_size"]

    def __init__(self, **kwargs):  # arguments must be passed as kwargs
        for k, val in kwargs.items():
            setattr(self, k, val)

        # check that all relevant configs are there
        annotations = {}
        for clas in type(self).mro():
            if hasattr(clas, "__annotations__"):
                annotations.update(clas.__annotations__)
        for k, k_type in annotations.items():
            assert hasattr(self, k), f"Key {k} with type {k_type} required for initialization."
            attr = getattr(self, k)
            if k in self.none_allowed:
                k_type = (k_type, type(None))
            assert isinstance(attr, k_type), f"Key {k} must be type {k_type}, provided={attr}."

        self.global_step = 0

    def build_optimizer(self, network: nn.Module) -> tuple[any, any]:
        """
        Require setting 'n_steps' and other related step configurations
        before building optimizer & lr_scheduler.
        """
        param_dict = {}
        for name, param in network.named_parameters():
            if param.requires_grad:
                opt_config = [self.weight_decay, self.init_lr]
                if self.no_wd_keys is not None and len(self.no_wd_keys) > 0:
                    if np.any([key in name for key in self.no_wd_keys]):
                        opt_config[0] = 0
                opt_key = json.dumps(opt_config)
                param_dict[opt_key] = param_dict.get(opt_key, []) + [param]

        net_params = []
        for opt_key, param_list in param_dict.items():
            wd, lr = json.loads(opt_key)
            net_params.append({"params": param_list, "weight_decay": wd, "lr": lr})

        optimizer = make_optimizer(net_params, self.optimizer_name, self.optimizer_params, self.init_lr)

        # build lr scheduler
        if self.lr_schedule_name == "cosine":
            decay_steps = self.lr_schedule_param.get("step", [self.n_steps-self.warmup_steps])
            decay_steps.sort()
            lr_scheduler = CosineLRwithWarmup(
                optimizer,
                self.warmup_steps,
                self.warmup_lr,
                decay_steps,
            )
        else:
            raise NotImplementedError

        return optimizer, lr_scheduler

    def build_loss(self,):
        return make_loss(self.loss)


    def update_global_step(self, step: int) -> None:
        """
        Update the global step count.
        """
        self.global_step = step
        Scheduler.PROGRESS = self.progress

    @property
    def progress(self) -> float:
        """
        Compute the current training progress as a fraction of total steps.
        """
        return self.global_step / self.n_steps

    def step(self) -> None:
        """
        Increment the global step count and update progress.
        """
        self.global_step += 1
        Scheduler.PROGRESS = self.progress

    def get_remaining_steps(self, current_step: int, post=True) -> int:
        """
        Get the remaining training steps from the current step.
        """
        return self.n_steps - current_step - int(post)

    def step_format(self, step: int) -> str:
        """
        Format the current step in a string with total step information.
        """
        step_format = f"%.{len(str(self.n_steps))}d"
        step_format = f"[{step_format}/{step_format}]"
        step_format = step_format % (step + 1, self.n_steps)
        return step_format




