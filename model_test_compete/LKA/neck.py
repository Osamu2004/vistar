import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class LKAneck(nn.Module):
    
    
    def forward_tensor(self, t1_input: dict,t2_input:dict):
        t1_output={}
        t2_output={}
        input[0] = self.patch_embed11(input[0])
        input[1] = self.patch_embed12(input[1])
        B, C, H, W = input[0].shape
        input[0] = input[0] + self._get_pos_embed(self.pos_embed1, self.num_patches1, H, W) 
        input[1] = input[1] + self._get_pos_embed(self.pos_embed1, self.num_patches1, H, W) 
        input = self.blocks1(input)
        t1_output["stage1"]=input[0]
        t2_output["stage1"]=input[1]

        input = self.patch_embed2(input)
        B, C, H, W = input[0].shape
        input[0] = input[0] + self._get_pos_embed(self.pos_embed2, self.num_patches2, H, W) 
        input[1] = input[1] + self._get_pos_embed(self.pos_embed2, self.num_patches2, H, W) 
        input[0], input[1] = self.FeatureEmbedding2(input[0], input[1]) 
        input = self.blocks2(input)
        t1_output["stage2"]=input[0]
        t2_output["stage2"]=input[1]

        input = self.patch_embed3(input)
        B, C, H, W = input[0].shape
        input[0] = input[0] + self._get_pos_embed(self.pos_embed3, self.num_patches3, H, W) 
        input[1] = input[1] + self._get_pos_embed(self.pos_embed3, self.num_patches3, H, W) 
        input[0], input[1] = self.FeatureEmbedding3(input[0], input[1]) 
        input = self.blocks3(input)
        t1_output["stage3"]=input[0]
        t2_output["stage3"]=input[1]

        input = self.patch_embed4(input)
        B, C, H, W = input[0].shape
        input[0] = input[0] + self._get_pos_embed(self.pos_embed4, self.num_patches4, H, W) 
        input[1] = input[1] + self._get_pos_embed(self.pos_embed4, self.num_patches4, H, W) 
        input[0], input[1] = self.FeatureEmbedding4(input[0], input[1]) 
        input = self.blocks4(input)
        t1_output["stage4"]=input[0]
        t2_output["stage4"]=input[1]
        return t1_output,t2_output
    
    def forward(self, batch_inputs: torch.Tensor, data_samples: Union[dict, tuple, list], mode='tensor'):
        data_samples = torch.stack(data_samples)
        if mode == 'tensor':
            return self.forward_tensor(batch_inputs)
        elif mode == 'predict':
            feats = self.forward_tensor(batch_inputs)
            predictions = torch.argmax(feats, 1)
            return feats, predictions
        elif mode == 'loss':
            feats = self.forward_tensor(batch_inputs)
            loss = self.criterion(feats, data_samples)
            return feats, dict(loss=loss)
