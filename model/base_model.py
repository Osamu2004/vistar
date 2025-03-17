from model.base_module import BaseModule

class BaseModel(BaseModule):
    def __init__(self,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        if data_preprocessor is None:
            data_preprocessor = dict(type='BaseDataPreprocessor')
        if isinstance(data_preprocessor, nn.Module):
            self.data_preprocessor = data_preprocessor
        elif isinstance(data_preprocessor, dict):
            self.data_preprocessor = MODELS.build(data_preprocessor)
        else:
            raise TypeError('data_preprocessor should be a `dict` or '
                            f'`nn.Module` instance, but got '
                            f'{type(data_preprocessor)}')
    @abstractmethod
    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]: