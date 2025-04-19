
import numpy as np
import albumentations as A
from apps.data_provider import DataProvider 

from cdcore.data_provider.base import BitemporalDataset
from apps.builder import make_augmentations
from apps.registry import DATAPROVIDER
from apps.cropper import DatasetCropper,ImgStats


class LevirCDDataset(BitemporalDataset):
    def change_onehot(self, changelabel):
        class_values = [0,255]
        label_out = np.zeros(changelabel.shape, dtype=changelabel.dtype)
        for idx, class_value in enumerate(class_values):
            label_out[changelabel == class_value] = idx
        return label_out


@DATAPROVIDER.register("s2looking")
class S2lookingDataProvider(DataProvider):
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
        self.main_classes = 2
        self.samples = [
            "image",  # 无变化
            "t2_image",     
            "mask",# 有变化
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
            raise ValueError("augmentation config cannot be None.")
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
                },
            )
        return main_transform, t1_transform, t2_transform
    
    def build_test_transform(self,config,t1_mean_std,t2_mean_std):
        if config is None:
            raise ValueError("test augmentation config cannot be None.")
        required_keys = ["shared", "t1", "t2"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key '{key}' in augmentation_config.")
        shared = make_augmentations(config.get("shared", []))
        t1_specific = make_augmentations(config.get("t1", None))
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
                },
            )
        return main_transform, t1_transform, t2_transform

    def build_datasets(self, only_test: bool = False) -> tuple[any, any, any]:
        root_dict = self.dataset_cropper.crop_all()
        t1_mean_std = self.imgstats.compute_mean_std(root_dir=f"{self.root}/train/Image1", output_dir=f"{self.root}/t1TrainsStats.yaml")
        t2_mean_std = self.imgstats.compute_mean_std(root_dir=f"{self.root}/train/Image2", output_dir=f"{self.root}/t2TrainsStats.yaml")
        
        train_pipeline = self.build_train_transform(self.data_aug.get("train"), t1_mean_std, t2_mean_std)
        valid_pipeline = self.build_valid_transform(self.data_aug.get("val"), t1_mean_std, t2_mean_std)
        test_pipeline = self.build_test_transform(self.data_aug.get("test"), t1_mean_std, t2_mean_std)

        # 仅在不是测试模式时，加载训练集和验证集
        if not only_test:
            train_dataset = LevirCDDataset(
                t1_image_root=root_dict["train"]["Image1"], 
                t2_image_root=root_dict["train"]["Image2"], 
                change_mask_root=root_dict["train"]["label"], 
                transform_pipelines=train_pipeline
            )
            
            val_dataset = LevirCDDataset(
                t1_image_root=root_dict["val"]["Image1"], 
                t2_image_root=root_dict["val"]["Image2"], 
                change_mask_root=root_dict["val"]["label"], 
                transform_pipelines=valid_pipeline
            )
        else:
            train_dataset = None
            val_dataset = None

        # 始终加载测试集
        test_dataset = LevirCDDataset(
            t1_image_root=root_dict["test"]["Image1"], 
            t2_image_root=root_dict["test"]["Image2"], 
            change_mask_root=root_dict["test"]["label"], 
            transform_pipelines=test_pipeline
        )

        # 返回数据集
        return train_dataset, val_dataset, test_dataset


