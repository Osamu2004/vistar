import torch
from typing import Dict, Optional
from model.base_module import BaseModule
# 假设模型类继承自BaseModel
class Segmentor(BaseModule):
    def __init__(self, init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)

    def slide_inference(self,input_dict, selected_keys, crop_size, stride):
        # 获取输入的形状
        B, _, H, W = input_dict[selected_keys[0]].shape
        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        out_channels = self.out_channels
        # 计算滑动窗口的步数
        h_grids = max(H - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(W - w_crop + w_stride - 1, 0) // w_stride + 1

        output_dict = {
            "main_output": torch.zeros((B, out_channels, H, W), device=input_dict[selected_keys[0]].device),
            "count_map": torch.zeros((B, 1, H, W), device=input_dict[selected_keys[0]].device)
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
                    patch_outputs = self.forward(patch_inputs)  # 或者直接使用 self(patch_inputs)

                # 将预测结果合并到输出张量
                output_dict["main_output"][:, :, y1:y2, x1:x2] += patch_outputs
                output_dict["count_map"][:, :, y1:y2, x1:x2] += 1
        assert (output_dict["count_map"] == 0).sum() == 0
        # 归一化结果
        output_dict["main_output"] /= output_dict["count_map"]

        # 移除 count_map
        output_dict.pop("count_map")
        return output_dict["main_output"]


    def whole_inference(self, input_dict, selected_keys):
        """
        Perform whole-image inference without using sliding windows.

        Args:
            input_dict (dict): A dictionary containing input tensors, e.g., {
                "image": (B, C, H, W),
                "t2_image": (B, C, H, W),
                "mask": (B, 1, H, W)  # Optional
            }
            selected_keys (list): List of keys to process, for example ["image", "t2_image"].

        Returns:
            Tensor: The output tensor with the same shape as the input (B, C_out, H, W).
        """

        # Process the whole image (no sliding window)
        with torch.no_grad():
            # Perform model inference using the entire input image
            patch_inputs = {key: input_dict[key] for key in selected_keys}
            patch_outputs = self.forward(patch_inputs)  # or directly use self(patch_inputs)

        # Assign the model outputs to the output dictionary

        return patch_outputs
