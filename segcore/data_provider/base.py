import torch
from os.path import splitext
from os import listdir
from skimage import io
from torch.utils.data import Dataset
import numpy as np
from apps.augment.augment import apply_totensor
from PIL import Image

class SegDataset(Dataset):
    def __init__(self, 
                 image_root, 
                 mask_root=None,
                 transform_pipelines=None):
        self.main_transform= transform_pipelines
        files = [file for file in listdir(image_root) if not file.startswith('.')]
        self.ids = sorted([splitext(file)[0] for file in files])
        extensions = set(splitext(file)[1] for file in files)
        if len(extensions) > 1:
            raise ValueError(f"Inconsistent extensions in T1 images: {extensions}")
        self.imagetype = extensions.pop()  
        self.image_root = image_root
        self.mask_root = mask_root


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # 获取文件 ID
        name = self.ids[idx]

        # 构造文件路径
        image_path = f"{self.image_root}/{name}{self.imagetype}"

        # 加载 T1 和 T2 图像
        image = Image.open(image_path)
        image = np.array(image).astype(np.uint8)
        # 加载变更掩码（如果提供）  
        label = None
        if self.mask_root:
            label_path = f"{self.mask_root}/{name}{self.imagetype}"
            label = Image.open(label_path).convert('L')
            label = np.array(label).astype(np.uint8)
            label = self.change_onehot(label)
            #changelabel = torch.tensor(changelabel, dtype=torch.float32)

        #print(image.shape,label.shape)
        sample = {
            "image": image,
            "mask": label,
        }

        # 应用数据增强（如果提供）
        if self.main_transform:
            transformed = self.main_transform(**sample)
            sample.update(transformed)  # 更新主图像和共享 mask 的增强结果
        return sample

    def change_onehot(self, changelabel):
        raise NotImplementedError

    def mask_onehot(self, masklabel):
        raise NotImplementedError


