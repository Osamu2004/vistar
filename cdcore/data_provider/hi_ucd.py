
from cdcore.data_provider.base import BitemporalDataset
import numpy as np
from skimage import io
from apps.augment.augment import apply_totensor
from apps.registry import DATAPROVIDER
from apps.data_provider import DataProvider 
from apps.cropper import DatasetCropper,ImgStats
import numpy as np
import albumentations as A
from apps.data_provider import DataProvider 

from cdcore.data_provider.base import BitemporalDataset
from apps.builder import make_augmentations
from apps.registry import DATAPROVIDER
from apps.cropper import DatasetCropper,ImgStats


class HiUCDDataset(BitemporalDataset):
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
        label_path = f"{self.change_mask_root}/{name}{self.imagetype}"
        label = io.imread(label_path)
        t1_mask = label[:, :, 0]  # First channel is T1
        t2_mask = label[:, :, 1]  # Second channel is T2
        changelabel = label[:, :, 2]  # Last channel is change

        sample = {
            "image": image1,
            "t2_image": image2,
            "mask": changelabel,
            "t1_mask":t1_mask,
            "t2_mask":t2_mask
        }

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


@DATAPROVIDER.register("hiucd")
class HiUCDDataProvider(DataProvider):
    name = "s2looking"

    def __init__(
        self,
        root: str,
        data_aug: dict or list[dict] or None,
        data_crop: dict,
        ############################
        train_batch_size: int,
        test_batch_size: int,
        n_worker=8,
        num_replicas: int or None = None,
        rank: int or None = None,
        train_ratio: float or None = None,
        drop_last: bool = False,
        num_threads: int = 4,
        only_test: bool = False,
    ):
        self.root = root
        self.data_aug = data_aug
        self.dataset_cropper = (
            DatasetCropper(root=root, data_crop=data_crop, num_threads=num_threads)
            if data_crop
            else None
        )
        self.imgstats = ImgStats()
        self.change_classes = 3
        self.semantic_classes = 10
        self.samples = [
            "image",  # 无变化
            "t2_image",     
            "mask",# 有变化
            "t1_mask",
            "t2_mask",
        ]

        super().__init__(
            train_batch_size = train_batch_size,
            test_batch_size = test_batch_size,
            n_worker = n_worker,
            num_replicas = num_replicas,
            rank = rank,
            train_ratio = train_ratio,
            drop_last = drop_last,
            only_test = only_test,
        )
    def build_train_transform(self,config,t1_mean_std,t2_mean_std):
        if config is None:
            raise ValueError("trsin augmentation config cannot be None.")
        required_keys = ["shared", "t1", "t2"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key '{key}' in augmentation_config.")
        shared = make_augmentations(config.get("shared", []))
        t1_specific = make_augmentations(config.get("t1", []))
        t2_specific = make_augmentations(config.get("t2", []))
        t1_specific.append(A.Normalize(mean=t1_mean_std["mean"], std=t1_mean_std["std"], max_pixel_value=255.0))
        t2_specific.append(A.Normalize(mean=t2_mean_std["mean"], std=t2_mean_std["std"], max_pixel_value=255.0))
        t1_transform = A.Compose(t1_specific)
        t2_transform = A.Compose(t2_specific)
        main_transform = A.Compose(
                shared,  
                additional_targets={
                    "t2_image": "image",
                    "mask": "mask",
                    "t1_mask":"mask",
                    "t2_mask":"mask",
                },
            )
        return main_transform, t1_transform, t2_transform
    

    def build_valid_transform(self,config,t1_mean_std,t2_mean_std):
        if config is None:
            raise ValueError("val augmentation config cannot be None.")
        required_keys = ["shared", "t1", "t2"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key '{key}' in augmentation_config.")
        shared = make_augmentations(config.get("shared", []))
        t1_specific = make_augmentations(config.get("t1", []))
        t2_specific = make_augmentations(config.get("t2", []))
        t1_specific.append(A.Normalize(mean=t1_mean_std["mean"], std=t1_mean_std["std"], max_pixel_value=255.0))
        t2_specific.append(A.Normalize(mean=t2_mean_std["mean"], std=t2_mean_std["std"], max_pixel_value=255.0))
        t1_transform = A.Compose(t1_specific)
        t2_transform = A.Compose(t2_specific)
        main_transform = A.Compose(
                shared,  
                additional_targets={
                    "t2_image": "image",
                    "mask": "mask",
                    "t1_mask":"mask",
                    "t2_mask":"mask",
                },
            )
        return main_transform, t1_transform, t2_transform
    

    def build_datasets(self, only_test: bool = False) -> tuple[any, any, any]:
        root_dict = self.dataset_cropper.crop_all()
        t1_mean_std = self.imgstats.compute_mean_std(root_dir=f"{self.root}/train/2018", output_dir=f"{self.root}/t1TrainsStats.yaml")
        t2_mean_std = self.imgstats.compute_mean_std(root_dir=f"{self.root}/train/2019", output_dir=f"{self.root}/t2TrainsStats.yaml")
        
        train_pipeline = self.build_train_transform(self.data_aug.get("train"), t1_mean_std, t2_mean_std)
        valid_pipeline = self.build_valid_transform(self.data_aug.get("val"), t1_mean_std, t2_mean_std)

        # 仅在不是测试模式时，加载训练集和验证集
        if not only_test:
            train_dataset = HiUCDDataProvider(
                t1_image_root=root_dict["train"]["2018"], 
                t2_image_root=root_dict["train"]["2019"], 
                change_mask_root=root_dict["train"]["2018_2019"], 
                transform_pipelines=train_pipeline
            )
            
            val_dataset = HiUCDDataProvider(
                t1_image_root=root_dict["val"]["2018"], 
                t2_image_root=root_dict["val"]["2019"], 
                change_mask_root=root_dict["val"]["2018_2019"], 
                transform_pipelines=valid_pipeline
            )
        else:
            train_dataset = None
            val_dataset = None
        test_dataset = None
        # 返回数据集
        return train_dataset, val_dataset, test_dataset