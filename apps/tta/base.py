import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from typing import List, Dict

class BaseTTAModel(nn.Module):
    """A model wrapper that applies test-time augmentation (TTA) using torchvision.transforms."""
    
    def __init__(self, model: nn.Module, scale_factors: List[float], flip: bool = True, selected_keys: List[str] = None):
        super().__init__()
        self.model = model
        self.scale_factors = scale_factors
        self.flip = flip
        self.selected_keys = selected_keys if selected_keys is not None else []
    
    def apply_tta(self, inputs: Dict[str, torch.Tensor]):
        """Applies TTA transformations to specified keys in the input dictionary."""
        augmented_inputs = []
        transforms = []

        orig_h, orig_w = inputs[self.selected_keys[0]].shape[-2:]

        for scale in self.scale_factors:
            new_h, new_w = int(orig_h * scale), int(orig_w * scale)
            
            aug_batch = {}
            for key, tensor in inputs.items():
                if key in self.selected_keys:
                    aug_batch[key] = F.resize(tensor, size=[new_h, new_w])
                else:
                    aug_batch[key] = tensor  # 不需要增强的键保持不变

            
            augmented_inputs.append(aug_batch)  
            transforms.append((scale, False, False))

            if self.flip:
                for flip_type, flip_func in [("h", F.hflip), ("v", F.vflip)]:
                    flipped_data = {}
                    for key, tensor in aug_batch.items():
                        if key in self.selected_keys:
                            flipped_data[key] = flip_func(tensor)
                        else:
                            flipped_data[key] = tensor  # 不变

                    augmented_inputs.append(flipped_data)
                    transforms.append((scale, flip_type == "h", flip_type == "v"))
                flipped_data = {}
                for key, tensor in aug_batch.items():
                    if key in self.selected_keys:
                        flipped_data[key] = F.vflip(F.hflip(tensor))
                    else:
                        flipped_data[key] = tensor  # 不变
                    augmented_inputs.append(flipped_data)
                    transforms.append((scale, True, True))
        
        return augmented_inputs, transforms
    
    def inverse_tta(self, predictions: List[torch.Tensor], transforms: List[tuple]):
        """Reverts TTA transformations and averages predictions."""
        inverse_predictions = []
        
        for pred, (scale, h_flip, v_flip) in zip(predictions, transforms):
            if h_flip:
                pred = F.hflip(pred)
            if v_flip:
                pred = F.vflip(pred)
            
            if scale != 1.0:
                inverse_scale = 1.0 / scale
                new_h, new_w = int(pred.shape[2] * inverse_scale), int(pred.shape[3] * inverse_scale)
                pred = F.resize(pred, size=[new_h, new_w])
            
            inverse_predictions.append(pred)
        return inverse_predictions
        #return torch.mean(torch.stack(inverse_predictions), dim=0)
    def fuse(self,inverse_predictions):
        preds = []
        for pred in inverse_predictions:
            if pred.shape[1] > 1:  # 多分类任务
                pred = torch.softmax(pred, dim=1)
            else:  # 二分类（单通道）
                pred = torch.sigmoid(pred)
            preds.append(pred)
        return torch.mean(torch.stack(preds), dim=0)
        
    def forward(self, inputs: Dict[str, torch.Tensor]):
        """Runs inference with TTA applied and merges the predictions."""
        augmented_inputs, transforms = self.apply_tta(inputs)
        predictions = [self.model(input_batch) for input_batch in augmented_inputs]
        inverse_predictions = self.inverse_tta(predictions, transforms)
        
        return self.fuse(inverse_predictions)
