# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

from model_test.CDformer.cdseg import (
    CBFCDSeg,
    cbf_seg_b1,
    cbf_seg_b2,
    cbf_seg_b3,

)
from model_test_compete.CDformer.cdseg import (
    cbf_seg_testb1,

)
from model_test_compete.LKA.cdseg import (
    lka_seg_lkab1,

)
from model_test_compete.LKA.cdseg import (
    lka_seg_lkab1,

)
from model_test_compete.SegNext.cdseg_uper import (
    segnext_seg_uperb1,

)
segnext_seg_uperb1
from model_test_compete.LKA.cdseg_uper import (
    lka_seg_lkauperb1,
    lka_seg_lkauperb2,
    lka_seg_lkauperb3,

)
from apps.utils.model import load_state_dict_from_file
from model_hky.segmentor import segmentor
__all__ = ["create_cd_model"]


REGISTERED_CLS_MODEL: dict[str, str] = {

}


def create_cd_model(name: str, pretrained=False, weight_url: str or None = None, **kwargs) -> CBFCDSeg:
    model_dict = {
        "b1": cbf_seg_b1,
        "b2": cbf_seg_b2,
        "b3": cbf_seg_b3,
        "testb1":cbf_seg_testb1,
        "lkab1":lka_seg_lkab1,
        "lkauperb1": lka_seg_lkauperb1,
        "lkauperb2": lka_seg_lkauperb2,
        "lkauperb3": lka_seg_lkauperb3,
        "segnextuperb1":segnext_seg_uperb1,
        "hky":segmentor
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
            model.load_state_dict(weight,strict = False)
    return model
