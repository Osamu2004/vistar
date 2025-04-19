import torch
import os
from inspect import signature
import torch.nn as nn

def build_kwargs_from_config(config: dict, target_func: callable):
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs

def val2list(x: list or tuple or any, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]

def load_state_dict_from_file(file: str, only_state_dict=True) -> dict[str, torch.Tensor]:
    file = os.path.realpath(os.path.expanduser(file))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(file, map_location=device)
    if only_state_dict and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint

from torch.utils.model_zoo import load_url

def load_state_dict_from_url(url: str,map_location ,only_state_dict=True) -> dict[str, torch.Tensor]:
    checkpoint = load_url(url, map_location=map_location,progress=True)
    if only_state_dict and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint

def is_parallel(model: nn.Module) -> bool:
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))

def list_join(x: list, sep="\t", format_str="%s") -> str:
    return sep.join([format_str % val for val in x])
def list_sum(x: list) -> any:
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def list_mean(x: list) -> any:
    return list_sum(x) / len(x)


def load_checkpoint_with_prefix(prefix, filename, map_location=None):
    """Load partial pretrained model with specific prefix.

    Args:
        prefix (str): The prefix of sub-module.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`.
            Defaults to None.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    checkpoint = torch.load(filename, map_location=map_location)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    state_dict = {
        k[prefix_len:]: v
        for k, v in state_dict.items() if k.startswith(prefix)
    }

    assert state_dict, f'{prefix} is not in the pretrained model'
    return state_dict