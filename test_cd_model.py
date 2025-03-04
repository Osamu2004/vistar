
import argparse
import math
import os
import torch
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from apps.utils.model import build_kwargs_from_config
from apps.builder import make_dataprovider
from cdcore.registry import register_cd_all
from apps.utils.misc import parse_unknown_args
from apps import setup
from cdcore.cd_model_zoo import create_cd_model
import numpy as np
import cv2
from PIL import Image
from apps.tta.base import TTAModel
parser = argparse.ArgumentParser()
parser.add_argument("config", metavar="FILE", help="config file")
from apps.utils.model import is_parallel, load_state_dict_from_file
from apps.utils.ema import EMA
CLASS_COLORS = {
    0: (255, 255, 255),  # 背景，白色
    1: (70, 181, 121),  # Intact，绿色
    2: (228, 189, 139),  # Damaged，浅棕色
    3: (182, 70, 69),    # Destroyed，红色
}

import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

def tensor_split_and_infer(model, input_dict, selected_keys, crop_size=(256, 256), stride=(128, 128),out_channels=4):
    """
    适配字典格式数据，将指定 keys 进行切割后输入模型推理，最后拼接回原尺寸。

    Args:
        model (torch.nn.Module): 需要执行的模型。
        input_dict (dict): 包含多个输入张量的字典，形如：
            {
                "image": (B, C, H, W),
                "t2_image": (B, C, H, W),
                "mask": (B, 1, H, W) # 可选
            }
        selected_keys (list): 需要进行切割和推理的键，例如 `["image", "t2_image"]`。
        crop_size (tuple): (height, width)，裁剪块大小，默认 (256, 256)。
        stride (tuple): (step_y, step_x)，滑动步长，默认 (128, 128)。
        batch_size (int): 多少个 patch 组成一个 batch 进行推理，提高推理效率。

    Returns:
        dict: 包含模型输出的字典，与输入分辨率一致：
            {
                "main_output": (B, C_out, H, W),
                "extra_output": (B, C_extra, H, W) # 若模型有额外输出
            }
    """
    # 获取输入的形状
    B, _, H, W = input_dict[selected_keys[0]].shape
    h_stride, w_stride = stride
    h_crop, w_crop = crop_size
    
    # 计算滑动窗口的步数
    h_grids = max(H - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(W - w_crop + w_stride - 1, 0) // w_stride + 1
    

    out_channels = out_channels
    
    output_dict = {
        "main_output": torch.zeros((B, out_channels, H, W), dtype=torch.float32, device=input_dict[selected_keys[0]].device),
        "count_map": torch.zeros((B, 1, H, W), dtype=torch.float32, device=input_dict[selected_keys[0]].device)
    }
    
    # 遍历滑动窗口
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, H)
            x2 = min(x1 + w_crop, W)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            
            # 取出当前窗口的 patch
            patch_inputs = {key: input_dict[key][:, :, y1:y2, x1:x2] for key in selected_keys}
            
            # 进行模型推理
            with torch.no_grad():
                patch_outputs = model(patch_inputs)  # 模型可能返回多个输出
            
            # 将预测结果合并到输出张量
            output_dict["main_output"][:, :, y1:y2, x1:x2] += patch_outputs
            output_dict["count_map"][:, :, y1:y2, x1:x2] += 1
            
            # 处理额外输出
    
    # 归一化结果
    for key in ["main_output"]:
        if key in output_dict:
            output_dict[key] /= torch.clamp(output_dict["count_map"], min=1.0)
    
    # 移除 count_map
    output_dict.pop("count_map")
    return output_dict["main_output"]


def save_image_with_name(image, name, output_dir):
    """
    将预测的 NumPy 数组保存为 uint8 PNG 文件。

    Args:
        image (np.ndarray): 预测的图像数组。
        name (str): 保存的文件名。
        output_dir (str): 保存目录。
    """
    os.makedirs(output_dir, exist_ok=True)

    # **确保 `name` 变量包含 `.png`**
    if not name.lower().endswith(".png"):
        name += ".png"

    output_path = os.path.join(output_dir, name)

    # **确保图像数据类型为 uint8**
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # **转换 NumPy 数组为 PIL 图像**
    image_pil = Image.fromarray(image)

    # **检查输出路径**
    print(f"⚡ 正在保存: {output_path}")

    # **保存图像**
    image_pil.save(output_path, format="PNG")
    print(f"✅ 图像已保存: {output_path}")

def main():
    register_cd_all()

    args = parser.parse_args()

    args, opt = parser.parse_known_args()
    opt = parse_unknown_args(opt)
    config = setup.setup_exp_config(args.config, recursive=True, opt_args=opt)
    config["data_provider"]["only_test"] = True
    data_provider = setup.setup_data_provider(config, is_distributed=False)
    model = create_cd_model(config["net_config"]["name"],pretrained = True,dataset = config['data_provider'].get('type'),
                            weight_url=r"E:\zzy\mmchange0.2\dfc25_track2_segnextffnb1\checkpoint\checkpoint.pt",
                            )
    #model = torch.nn.DataParallel(model).cuda()
    #print(data_provider.test)

    checkpoint = load_state_dict_from_file(r"E:\zzy\mmchange0.2\dfc25_track2_segnextffnb1\checkpoint\checkpoint.pt", only_state_dict = False)
    model_ema = EMA(model, 0.9998)
    model_ema.load_state_dict(checkpoint["ema"])
    
    model = model_ema.shadows.cuda().eval() 
    #model = model.cuda().eval() 
    model = TTAModel(model=model, scale_factors=[1.0], flip=True,tta_keys=["image","t2_image"])

    with torch.no_grad():
        with tqdm(
                total=len(data_provider.test),
                desc=f"Validate Step #",
            ) as t:
            for samples in data_provider.test:
                images = samples.get("image", None)
                t2_images = samples.get("t2_image", None)
                labels = samples.get("mask", None)
                t1_mask = samples.get("t1_mask", None)
                t2_mask = samples.get("t2_mask", None)
                names =  samples.get("name", "unnamed")


                    # 数据转到 GPU
                images = images.cuda()
                t2_images = t2_images.cuda()
                if t1_mask is not None:
                    t1_mask = t1_mask.long().cuda()
                if t2_mask is not None:
                    t2_mask = t2_mask.long().cuda()

                    # 模型输入
                model_input = {"image": images, "t2_image": t2_images}
                output = tensor_split_and_infer(model,model_input,["image","t2_image"], crop_size=(256, 256), stride=(128, 128))
                #output = model(model_input)

                    # 模型可能返回多个输出时进行解包
                predictions = torch.argmax(output, dim=1)


                    # 保存图像
                for i, name in enumerate(names):  # names 中包含每个样本的名字
                    pred_image = predictions.cpu().numpy()[i]  # 获取当前样本的预测结果

                    # 创建一个空的 RGB 图像
                    color_image = np.zeros((pred_image.shape[0], pred_image.shape[1], 3), dtype=np.uint8)

                    # 为每个类别分配颜色
                    for label, color in CLASS_COLORS.items():
                        color_image[pred_image == label] = color

                    # 保存图像
                    save_image_with_name(pred_image, name, output_dir=r"E:\zzy\mmchange0.2\dfc25_track2_segnextffnb1\raw_image_tta")
                    save_image_with_name(color_image, name, output_dir=r"E:\zzy\mmchange0.2\dfc25_track2_segnextffnb1\color_image_tta")
                t.update(1)  # 更新进度条

if __name__ == "__main__":
    main()


