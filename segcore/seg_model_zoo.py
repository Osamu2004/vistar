# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

from model_test.CDformer.cdseg import (
    CBFCDSeg,
    cbf_seg_b1,
    cbf_seg_b2,
    cbf_seg_b3,

)
from model_pretrain.segmentor import (
    segmentor_convnext,
    segmentor_inception_next,
    segmentor_mambaout,

)

from apps.utils.model import load_state_dict_from_file


__all__ = ["create_seg_model"]


REGISTERED_CLS_MODEL: dict[str, str] = {

}


def create_seg_model(name: str, pretrained=False, weight_url: str or None = None, **kwargs) -> CBFCDSeg:
    model_dict = {
        "pretrainconvnext":segmentor_convnext,
        "pretraininception":segmentor_inception_next,
        "pretrainmambaout":segmentor_mambaout
    }

    model_id = name.split("-")[1]
    if model_id not in model_dict:
        raise ValueError(f"Do not find {name} in the model zoo. List of models: {list(model_dict.keys())}")
    else:
        model = model_dict[model_id](**kwargs)

    if pretrained:
        weight_url = weight_url
        if weight_url is None:
            raise ValueError(f"Do not find the pretrained weight of {name}.")
        else:
            weight = load_state_dict_from_file(weight_url)
            model.load_state_dict(weight)
    return model
