

import numpy as np
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from apps.data_provider import DataProvider 
from apps.augment.augment import apply_totensor
from segcore.data_provider.base import SegDataset
from apps.builder import make_augmentations
from apps.registry import DATAPROVIDER
from apps.cropper import DatasetCropper,ImgStats
from torch.utils.data import ConcatDataset
from skimage import io
class WhumixDataset(SegDataset):
    def change_onehot(self, label):
        class_values = [0,255]
        label_out = np.zeros(label.shape, dtype=label.dtype)
        for idx, class_value in enumerate(class_values):
            label_out[label == class_value] = idx
        return label_out

class Xview2_Dataset(SegDataset):
    def change_onehot(self, label):
        label_out = np.zeros(label.shape, dtype=label.dtype)
        label_out[label != 0] = 1  # 将所有非零值置为 1，零值保持为 0
        return label_out
    def __getitem__(self, idx):
        # 获取文件 ID
        name = self.ids[idx]

        # 构造文件路径
        image_path = f"{self.image_root}/{name}{self.imagetype}"

        # 加载 T1 和 T2 图像
        image = io.imread(image_path)

        # 加载变更掩码（如果提供）
        label = None
        if self.mask_root:
            label_path = f"{self.mask_root}/{name}_target{self.imagetype}"
            label = io.imread(label_path, as_gray=True)
            label = self.change_onehot(label)
            #changelabel = torch.tensor(changelabel, dtype=torch.float32)


        sample = {
            "image": image,
            "mask": label,
        }

        # 应用数据增强（如果提供）
        if self.main_transform:
            transformed = self.main_transform(**sample)
            sample.update(transformed)  # 更新主图像和共享 mask 的增强结果

        return sample
    def check_ids(self):
        # 动态计算 T2 和其他 ID 列表
        pass

from albumentations.pytorch import ToTensorV2
@DATAPROVIDER.register("pretrain")
class WhumixDataProvider(DataProvider):
    name = "pretrain"

    def __init__(
        self,
        root: str,
        data_aug: dict or list[dict] or None,
        data_crop: dict,
        ############################
        train_batch_size: int,
        test_batch_size: int,
        valid_size: int or float or None = None,
        n_worker=8,
        num_replicas: int or None = None,
        rank: int or None = None,
        train_ratio: float or None = None,
        drop_last: bool = False,
        num_threads: int = 4,
    ):
        self.root = root
        self.data_aug = data_aug
        self.dataset_cropper = (
            DatasetCropper(root=root, data_crop=data_crop, num_threads=num_threads)
            if data_crop
            else None
        )
        self.imgstats = ImgStats()
        self.main_classes = 2
        super().__init__(
            train_batch_size,
            test_batch_size,
            valid_size,
            n_worker,
            num_replicas,
            rank,
            train_ratio,
            drop_last,
        )
    def build_train_transform(self,config,t1_mean_std):
        if config is None:
            raise ValueError("train augmentation config cannot be None.")
        required_keys = ["image"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key '{key}' in augmentation_config.")
        image = make_augmentations(config.get("image", []))
        image.append(A.Normalize(mean=t1_mean_std["mean"], std=t1_mean_std["std"], max_pixel_value=255.0))
        image.append(ToTensorV2())
        
        main_transform = A.Compose(
                image,  
                additional_targets={
                    "mask": "mask",
                },
            )
        return main_transform


    def build_datasets(self, only_test: bool = False) -> tuple[any, any, any]:
        root_dict = self.dataset_cropper.crop_all()
        mean_std = self.imgstats.compute_mean_std(root_dir = f"{self.root}/mmbuilding/rgb" , output_dir=f"{self.root}/TrainsStats_rgb.yaml" )
        train_pipeline = self.build_train_transform(self.data_aug.get("train"),mean_std)

        if not only_test:
            train_dataset = WhumixDataset(image_root=root_dict["mmbuilding"]["rgb"], mask_root=root_dict["mmbuilding"]["label"], transform_pipelines=train_pipeline)
        else:
            train_dataset = None
        val_dataset = None
        test_dataset = None

        return train_dataset, val_dataset, test_dataset