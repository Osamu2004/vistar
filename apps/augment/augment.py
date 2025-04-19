import albumentations as A
from apps.registry import AUGMENTATION
import cv2
# 随机裁剪
def string2tuple(string):
    # 去掉括号，并分割字符串
    string = string.strip('()')
    # 将数字字符串按逗号分开，转换为浮动类型并返回元组
    return tuple(map(float, string.split(',')))

@AUGMENTATION.register("random_crop")
def random_crop(height=256, width=256, p=1.0):
    return A.RandomCrop(height=height, width=width,pad_if_needed=True, p=p,border_mode=cv2.BORDER_REFLECT)

@AUGMENTATION.register("flip")
def flip(horizontal=True, vertical=False):
    transforms = []
    if horizontal:
        transforms.append(A.HorizontalFlip(p=0.5))
    if vertical:
        transforms.append(A.VerticalFlip(p=0.5))
    return A.Compose(transforms)

# 颜色抖动
@AUGMENTATION.register("color_jitter")
def color_jitter(brightness=(0.8,1.2), contrast=(0.8,1.2), saturation=(0.8,1.2), hue=(-0.5,0.5), p=1.0):
    return A.ColorJitter(brightness=string2tuple(brightness), contrast=string2tuple(contrast), saturation=string2tuple(saturation), hue=string2tuple(hue), p=p)

# 高斯噪声
@AUGMENTATION.register("gauss_noise")
def gauss_noise(std_range=(0.01,0.1), mean_range=(0,0), p=0.5):
    return A.GaussNoise(std_range=string2tuple(std_range), mean_range=string2tuple(mean_range), p=p)

@AUGMENTATION.register("rotate")
def rotate(limit=(-90,90), p=0.5):
    return A.Rotate(limit=string2tuple(limit), p=p,border_mode=cv2.BORDER_REFLECT_101)

@AUGMENTATION.register("random_scale")
def rotate(scale_limit=(-0.1,0.1), p=0.5):
    return A.RandomScale(scale_limit=string2tuple(scale_limit), p=p)

@AUGMENTATION.register("perspective")
def perspective(scale=(0.05, 0.1), p=0.5, keep_size=False):
    return A.Perspective(scale=string2tuple(scale), p=p, keep_size=keep_size)

@AUGMENTATION.register("coarse_dropout")
def coarse_dropout(num_holes_range=(1, 2), hole_height_range=(0.1, 0.4), hole_width_range=(0.1, 0.4), p=1.0):
    return A.CoarseDropout(
        num_holes_range=string2tuple(num_holes_range),
        hole_height_range=string2tuple(hole_height_range),
        hole_width_range=string2tuple(hole_width_range),
        p=p
    )



@AUGMENTATION.register("noop")
def noop():
    return A.NoOp()

from albumentations.pytorch import ToTensorV2
from albumentations import Compose

def apply_totensor(sample):
    """
    将样本中的所有字段转换为张量。
    支持可选字段 t1_mask 和 t2_mask。

    Args:
        sample (dict): 包含多个字段的字典，结构如下：
            {
                "image": image1,
                "t2_image": image2,
                "mask": changelabel,
                "t1_mask": optional,
                "t2_mask": optional
            }

    Returns:
        dict: 转换后的样本字典，其中所有字段都被转换为 PyTorch 张量。
    """
    # 默认值处理
    sample.setdefault("image", None)
    sample.setdefault("t2_image", None)
    sample.setdefault("mask", None)
    sample.setdefault("t1_mask", None)
    sample.setdefault("t2_mask", None)

    # 定义转换器，明确声明额外的目标字段
    to_tensor = Compose(
        [ToTensorV2()],
        additional_targets={
            "t2_image": "image",  # t2_image 被视为图像
            "t1_mask": "mask",    # t1_mask 被视为掩码
            "t2_mask": "mask"     # t2_mask 被视为掩码
        }
    )

    # 检查必需字段是否存在且非空
    for key in ["image", "t2_image"]:
        if sample[key] is None:
            raise ValueError(f"Key {key} is missing or has a None value in the sample.")

    # 准备 Albumentations 的输入字典
    alb_inputs = {
        "image": sample["image"],
        "t2_image": sample["t2_image"],
    }

    # 添加可选字段（t1_mask 和 t2_mask）到输入字典
    if sample["mask"] is not None:
        alb_inputs["mask"] = sample["mask"]
    if sample["t1_mask"] is not None:
        alb_inputs["t1_mask"] = sample["t1_mask"]
    if sample["t2_mask"] is not None:
        alb_inputs["t2_mask"] = sample["t2_mask"]

    # 使用 Albumentations 转换所有字段
    tensor_sample = to_tensor(**alb_inputs)

    # 返回转换后的字段
    return {
        key: tensor_sample[key] for key in sample.keys() if key in tensor_sample
    }


