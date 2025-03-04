from apps.registry import AUGMENTATION,LOSS,DATAPROVIDER,OPT


def make_dataprovider(config):
    """
    根据配置动态返回数据提供器类。

    Args:
        config (dict): 数据提供器配置，包含以下内容：
            - 'type' (str): 数据提供器的名称（必须已注册到 DATAPROVIDER 注册表）。

    Returns:
        provider_class (type): 数据提供器类。
    """
    # 获取数据提供器类型
    provider_type = config.get("type")
    
    # 检查数据提供器类型是否已注册
    if provider_type not in DATAPROVIDER:
        raise ValueError(f"DataProvider type '{provider_type}' is not registered in DATAPROVIDER registry.")
    
    # 返回数据提供器类
    provider_class = DATAPROVIDER[provider_type]
    return provider_class

def make_augmentations(configs):
    """
    根据配置动态生成数据增强列表。

    Args:
        configs (list[dict]): 数据增强配置列表，每个配置包含以下内容：
            - 'type' (str): 数据增强模块的名称（必须已注册到 AUGMENTATION 注册表）。
            - 'params' (dict): 数据增强模块的参数。
    
    Returns:
        augmentations (list): 数据增强操作的列表。
    """
    if not configs:
        return []
    augmentations = []

    for config in configs:
        # 获取数据增强类型
        augmentation_type = config.get("type")
        
        # 检查增强类型是否已注册
        if augmentation_type not in AUGMENTATION:
            raise ValueError(f"Augmentation type '{augmentation_type}' is not registered in AUGMENTATION registry.")
        
        # 根据配置中的参数生成数据增强对象
        augmentation = AUGMENTATION[augmentation_type](**config.get("params", {}))
        augmentations.append(augmentation)

    return augmentations



def make_loss(configs):
    """
    根据配置动态生成损失函数的加权组合。

    Args:
        configs (dict): 损失函数的配置字典，包括以下内容：
            - 'type1', 'type2', ... (str): 损失函数名称（必须已注册到 LOSS 注册表）。
            - 'params1', 'params2', ... (dict): 损失函数的参数。
            - 'weight1', 'weight2', ... (float): 损失函数的权重。
    
    Returns:
        combined_loss_fn (function): 一个计算加权损失的函数。
    """
    # 存储所有损失函数及其权重
    loss_fns = []
    weights = []
    print(configs)

    # 动态解析配置中的损失函数
    index = 1
    while f"type{index}" in configs:
        loss_type = configs[f"type{index}"]
        params = configs.get(f"params{index}", {})
        weight = configs.get(f"weight{index}", 1.0)

        # 检查损失函数是否已注册
        if loss_type not in LOSS:
            raise ValueError(f"Loss type '{loss_type}' is not registered in LOSS registry.")

        # 获取损失函数并存储
        loss_fns.append(LOSS[loss_type](**params))
        weights.append(weight)

        index += 1
    if "OHEM" in configs:
        print("OHEM is used")
        loss_fn_ohem = []
        index = 0
        for loss_fn in loss_fns:
            index = index +1
            if configs.get(f"wrapper{index}", False):
                print(1,loss_fn)
                ohem_loss = LOSS["OHEM"](loss_fn=loss_fn,
                                 thresh=configs["OHEM"]["thresh"],
                                 min_kept=configs["OHEM"]["min_kept"],)
                loss_fn_ohem.append(ohem_loss)
            else:
                print(2,loss_fn)
                loss_fn_ohem.append(loss_fn)
        def combined_loss_ohem(predictions, targets):
            total_loss = 0.0
            for loss_fn, weight in zip(loss_fn_ohem, weights):
                total_loss += weight * loss_fn(predictions, targets)
            return total_loss
        return combined_loss_ohem

    def combined_loss(predictions, targets):
        total_loss = 0.0
        for loss_fn, weight in zip(loss_fns, weights):
            total_loss += weight * loss_fn(predictions, targets)
        return total_loss


    return combined_loss


def make_optimizer(net_params,optimizer_name,optimizer_params, init_lr):
    """
    根据配置动态创建优化器。

    Args:
        optimizer_name (str): 优化器名称。
        net_params: 模型参数。
        init_lr (float): 初始学习率。
        optimizer_params (dict): 其他优化器参数。

    Returns:
        torch.optim.Optimizer: 优化器实例。
    """
    print(f"Registered optimizers: {list(OPT.keys())}")
    if optimizer_name not in OPT:
        raise ValueError(f"Optimizer '{optimizer_name}' is not registered.")
    return OPT[optimizer_name](net_params, init_lr, **(optimizer_params or {}))


'''
def make_callback(config):
    callback_type = config['type']
    if callback_type in registry.CALLBACK:
        callback = registry.CALLBACK[callback_type](**config['params'])
    else:
        raise ValueError('{} is not support now.'.format(callback_type))
    return callback


def make_optimizer(config, params):
    opt_type = config['type']
    if opt_type in registry.OPT:
        opt = registry.OPT[opt_type](params=params, **config['params'])
        opt.er_config = config
    else:
        raise ValueError('{} is not support now.'.format(opt_type))
    return opt


def make_learningrate(config):
    lr_type = config['type']
    if lr_type in registry.LR:
        lr_module = registry.LR[lr_type]
        return lr_module(**config['params'])
    else:
        raise ValueError('{} is not support now.'.format(lr_type))


def make_dataloader(config):
    dataloader_type = config['type']
    if dataloader_type in registry.DATALOADER:
        data_loader = registry.DATALOADER[dataloader_type](config['params'])
    elif dataloader_type in registry.DATASET:
        dataset = registry.DATASET[dataloader_type](config['params'])
        data_loader = dataset.to_dataloader()
    else:
        raise ValueError('{} is not support now.'.format(dataloader_type))

    return data_loader


def make_model(config):
    from ever.interface import ERModule
    from torch.nn import Module
    model_type = config['type']
    if model_type in registry.MODEL:
        if issubclass(registry.MODEL[model_type], ERModule):
            model = registry.MODEL[model_type](config['params'])
        elif issubclass(registry.MODEL[model_type], Module):
            model = registry.MODEL[model_type](**config['params'])
        else:
            raise ValueError(f'unsupported model class: {registry.MODEL[model_type]}')
    else:
        raise ValueError(
            '{} is not support now. This model seems not to be registered via @er.registry.MODEL.register()'.format(model_type))
    return model
'''