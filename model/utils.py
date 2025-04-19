def load_pretrained(
        model: nn.Module,
        pretrained_cfg: Optional[Dict] = None,
        num_classes: int = 1000,
        in_chans: int = 3,
        filter_fn: Optional[Callable] = None,
        strict: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
):
    """ Load pretrained checkpoint

    Args:
        model: PyTorch module
        pretrained_cfg: Configuration for pretrained weights / target dataset
        num_classes: Number of classes for target model. Will adapt pretrained if different.
        in_chans: Number of input chans for target model. Will adapt pretrained if different.
        filter_fn: state_dict filter fn for load (takes state_dict, model as args)
        strict: Strict load of checkpoint
        cache_dir: Override model checkpoint cache dir for this load
    """
    pretrained_cfg = pretrained_cfg or getattr(model, 'pretrained_cfg', None)
    if not pretrained_cfg:
        raise RuntimeError("Invalid pretrained config, cannot load weights. Use `pretrained=False` for random init.")

    load_from, pretrained_loc = _resolve_pretrained_source(pretrained_cfg)
    if load_from == 'state_dict':
        _logger.info(f'Loading pretrained weights from state dict')
        state_dict = pretrained_loc  # pretrained_loc is the actual state dict for this override
    elif load_from == 'file':
        _logger.info(f'Loading pretrained weights from file ({pretrained_loc})')
        if pretrained_cfg.get('custom_load', False):
            model.load_pretrained(pretrained_loc)
            return
        else:
            state_dict = load_state_dict(pretrained_loc)
    elif load_from == 'url':
        _logger.info(f'Loading pretrained weights from url ({pretrained_loc})')
        if pretrained_cfg.get('custom_load', False):
            pretrained_loc = download_cached_file(
                pretrained_loc,
                progress=_DOWNLOAD_PROGRESS,
                check_hash=_CHECK_HASH,
                cache_dir=cache_dir,
            )
            model.load_pretrained(pretrained_loc)
            return
        else:
            try:
                state_dict = load_state_dict_from_url(
                    pretrained_loc,
                    map_location='cpu',
                    progress=_DOWNLOAD_PROGRESS,
                    check_hash=_CHECK_HASH,
                    weights_only=True,
                    model_dir=cache_dir,
                )
            except TypeError:
                state_dict = load_state_dict_from_url(
                    pretrained_loc,
                    map_location='cpu',
                    progress=_DOWNLOAD_PROGRESS,
                    check_hash=_CHECK_HASH,
                    model_dir=cache_dir,
                )
    elif load_from == 'hf-hub':
        _logger.info(f'Loading pretrained weights from Hugging Face hub ({pretrained_loc})')
        if isinstance(pretrained_loc, (list, tuple)):
            custom_load = pretrained_cfg.get('custom_load', False)
            if isinstance(custom_load, str) and custom_load == 'hf':
                load_custom_from_hf(*pretrained_loc, model, cache_dir=cache_dir)
                return
            else:
                state_dict = load_state_dict_from_hf(*pretrained_loc, cache_dir=cache_dir)
        else:
            state_dict = load_state_dict_from_hf(pretrained_loc, weights_only=True, cache_dir=cache_dir)
    else:
        model_name = pretrained_cfg.get('architecture', 'this model')
        raise RuntimeError(f"No pretrained weights exist for {model_name}. Use `pretrained=False` for random init.")

    if filter_fn is not None:
        try:
            state_dict = filter_fn(state_dict, model)
        except TypeError as e:
            # for backwards compat with filter fn that take one arg
            state_dict = filter_fn(state_dict)

    input_convs = pretrained_cfg.get('first_conv', None)
    if input_convs is not None and in_chans != 3:
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = input_conv_name + '.weight'
            try:
                state_dict[weight_name] = adapt_input_conv(in_chans, state_dict[weight_name])
                _logger.info(
                    f'Converted input conv {input_conv_name} pretrained weights from 3 to {in_chans} channel(s)')
            except NotImplementedError as e:
                del state_dict[weight_name]
                strict = False
                _logger.warning(
                    f'Unable to convert pretrained {input_conv_name} weights, using random init for this layer.')

    classifiers = pretrained_cfg.get('classifier', None)
    label_offset = pretrained_cfg.get('label_offset', 0)
    if classifiers is not None:
        if isinstance(classifiers, str):
            classifiers = (classifiers,)
        if num_classes != pretrained_cfg['num_classes']:
            for classifier_name in classifiers:
                # completely discard fully connected if model num_classes doesn't match pretrained weights
                state_dict.pop(classifier_name + '.weight', None)
                state_dict.pop(classifier_name + '.bias', None)
            strict = False
        elif label_offset > 0:
            for classifier_name in classifiers:
                # special case for pretrained weights with an extra background class in pretrained weights
                classifier_weight = state_dict[classifier_name + '.weight']
                state_dict[classifier_name + '.weight'] = classifier_weight[label_offset:]
                classifier_bias = state_dict[classifier_name + '.bias']
                state_dict[classifier_name + '.bias'] = classifier_bias[label_offset:]

    load_result = model.load_state_dict(state_dict, strict=strict)
    if load_result.missing_keys:
        _logger.info(
            f'Missing keys ({", ".join(load_result.missing_keys)}) discovered while loading pretrained weights.'
            f' This is expected if model is being adapted.')
    if load_result.unexpected_keys:
        _logger.warning(
            f'Unexpected keys ({", ".join(load_result.unexpected_keys)}) found while loading pretrained weights.'
            f' This may be expected if model is being adapted.')

