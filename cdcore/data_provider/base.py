import torch
from os.path import splitext
from os import listdir
from skimage import io
from torch.utils.data import Dataset
import numpy as np
from apps.augment.augment import apply_totensor

class BitemporalDataset(Dataset):
    def __init__(self, 
                 t1_image_root, 
                 t2_image_root,
                 change_mask_root=None,
                 t1_mask_root=None,
                 t2_mask_root=None,
                 transform_pipelines=None):
        self.main_transform, self.t1_transform, self.t2_transform = transform_pipelines
        files = [file for file in listdir(t1_image_root) if not file.startswith('.')]
        self.t1_ids = sorted([splitext(file)[0] for file in files])
        extensions = set(splitext(file)[1] for file in files)
        if len(extensions) > 1:
            raise ValueError(f"Inconsistent extensions in T1 images: {extensions}")
        self.imagetype = extensions.pop()  
        self.t1_image_root = t1_image_root
        self.t2_image_root = t2_image_root
        self.change_mask_root = change_mask_root
        self.t1_mask_root = t1_mask_root
        self.t2_mask_root = t2_mask_root

        self.check_ids()

    def __len__(self):
        return len(self.t1_ids)

    def __getitem__(self, idx):
        # 获取文件 ID
        name = self.t1_ids[idx]

        # 构造文件路径
        image1_path = f"{self.t1_image_root}/{name}{self.imagetype}"
        image2_path = f"{self.t2_image_root}/{name}{self.imagetype}"

        # 加载 T1 和 T2 图像
        image1 = io.imread(image1_path)
        image2 = io.imread(image2_path)

        # 加载变更掩码（如果提供）
        changelabel = None
        if self.change_mask_root:
            label_path = f"{self.change_mask_root}/{name}{self.imagetype}"
            changelabel = io.imread(label_path)
            changelabel = self.change_onehot(changelabel)
            #changelabel = torch.tensor(changelabel, dtype=torch.float32)

        # 加载 T1 和 T2 掩码（如果提供）
        t1_mask = None
        t2_mask = None
        if self.t1_mask_root:
            t1_mask_path = f"{self.t1_mask_root}/{name}{self.imagetype}"
            t1_mask = io.imread(t1_mask_path)
            t1_mask = self.mask_onehot(t1_mask)
            #t1_mask = torch.tensor(t1_mask, dtype=torch.float32)

        if self.t2_mask_root:
            t2_mask_path = f"{self.t2_mask_root}/{name}{self.imagetype}"
            t2_mask = io.imread(t2_mask_path)
            t2_mask = self.mask_onehot(t2_mask)

        sample = {
            "image": image1,
            "t2_image": image2,
            "mask": changelabel,
        }
        if t1_mask is not None:
            sample["t1_mask"] = t1_mask
        if t2_mask is not None:
            sample["t2_mask"] = t2_mask

        # 应用数据增强（如果提供）
        if self.main_transform:
            transformed = self.main_transform(**sample)
            sample.update(transformed)  # 更新主图像和共享 mask 的增强结果

        # 分别应用 T1 和 T2 的特定增强
        if self.t1_transform:
            sample["image"] = self.t1_transform(image=sample["image"])["image"]
        if self.t2_transform:
            sample["t2_image"] = self.t2_transform(image=sample["t2_image"])["image"]
        sample = apply_totensor(sample)

        return sample

    def change_onehot(self, changelabel):
        raise NotImplementedError

    def mask_onehot(self, masklabel):
        raise NotImplementedError


    def check_ids(self):
        # 动态计算 T2 和其他 ID 列表
        t2_ids = sorted([splitext(file)[0] for file in listdir(self.t2_image_root) if not file.startswith('.')])

        if self.change_mask_root:
            change_ids = sorted([splitext(file)[0] for file in listdir(self.change_mask_root) if not file.startswith('.')])
        else:
            change_ids = None

        # 检查 self.t1_mask_root 是否为 None
        if self.t1_mask_root:
            t1_mask_ids = sorted([splitext(file)[0] for file in listdir(self.t1_mask_root) if not file.startswith('.')])
        else:
            t1_mask_ids = None

        # 检查 self.t2_mask_root 是否为 None
        if self.t2_mask_root:
            t2_mask_ids = sorted([splitext(file)[0] for file in listdir(self.t2_mask_root) if not file.startswith('.')])
        else:
            t2_mask_ids = None

        # 检查 T1 和 T2 图像 ID
        assert len(self.t1_ids) == len(t2_ids), "T1 and T2 IDs have different lengths."
        for t1, t2 in zip(self.t1_ids, t2_ids):
            assert t1 == t2, f"Mismatch in T1 and T2 IDs: {t1} vs {t2}"

        # 检查变更掩码 ID（如果提供）
        if change_ids:
            assert len(self.t1_ids) == len(change_ids), "T1 and change IDs have different lengths."
            for t1, change in zip(self.t1_ids, change_ids):
                assert t1 == change, f"Mismatch in T1 and change IDs: {t1} vs {change}"

        # 检查 T1 和 T2 掩码 ID（如果提供）
        if t1_mask_ids:
            assert len(self.t1_ids) == len(t1_mask_ids), "T1 and T1 mask IDs have different lengths."
            for t1, t1_mask in zip(self.t1_ids, t1_mask_ids):
                assert t1 == t1_mask, f"Mismatch in T1 and T1 mask IDs: {t1} vs {t1_mask}"

        if t2_mask_ids:
            assert len(t2_ids) == len(t2_mask_ids), "T2 and T2 mask IDs have different lengths."
            for t2, t2_mask in zip(t2_ids, t2_mask_ids):
                assert t2 == t2_mask, f"Mismatch in T2 and T2 mask IDs: {t2} vs {t2_mask}"