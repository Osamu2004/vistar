from apps.registry import OPT
import torch

@OPT.register("sgd")
def build_sgd_optimizer(net_params, init_lr, **kwargs):
    default_params = {"momentum": 0.9, "nesterov": True}
    default_params.update(kwargs)
    return torch.optim.SGD(net_params, init_lr, **default_params)

@OPT.register("adam")
def build_adam_optimizer(net_params, init_lr, **kwargs):
    default_params = {"betas": (0.9, 0.999), "eps": 1e-8, "amsgrad": False}
    default_params.update(kwargs)
    return torch.optim.Adam(net_params, init_lr, **default_params)

@OPT.register("adamw")
def build_adamw_optimizer(net_params, init_lr, **kwargs):
    default_params = {"betas": (0.9, 0.999), "eps": 1e-8, "amsgrad": False}
    default_params.update(kwargs)
    return torch.optim.AdamW(net_params, init_lr, **default_params)