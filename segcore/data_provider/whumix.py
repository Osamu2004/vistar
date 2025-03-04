

import numpy as np
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from apps.data_provider import DataProvider 

from segcore.data_provider.base import SegDataset
from apps.builder import make_augmentations
from apps.registry import DATAPROVIDER
from apps.cropper import DatasetCropper,ImgStats
from torch.utils.data import ConcatDataset

class WhumixDataset(SegDataset):
    def change_onehot(self, label):
        class_values = [255,0]
        label_out = np.zeros(label.shape, dtype=label.dtype)
        for idx, class_value in enumerate(class_values):
            label_out[label == class_value] = idx
        return label_out
    
@DATAPROVIDER.register("whumix")
class WhumixDataProvider(DataProvider):
    name = "whumix"

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
    def build_train_transform(self,config,t1_mean_std,t2_mean_std):
        if config is None:
            raise ValueError("train augmentation config cannot be None.")
        required_keys = ["image"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key '{key}' in augmentation_config.")
        image = make_augmentations(config.get("image", []))
        image.append(A.Normalize(mean=t1_mean_std["mean"], std=t1_mean_std["std"], max_pixel_value=255.0))
        main_transform = A.Compose(
                image,  
                additional_targets={
                    "mask": "mask",
                },
            )
        return main_transform

    def build_valid_transform(self,config,mean_std):
        if config is None:
            raise ValueError("train augmentation config cannot be None.")
        required_keys = ["image"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key '{key}' in augmentation_config.")
        image = make_augmentations(config.get("image", []))
        image.append(A.Normalize(mean=mean_std["mean"], std=mean_std["std"], max_pixel_value=255.0))
        main_transform = A.Compose(
                image,  
                additional_targets={
                    "mask": "mask",
                },
            )
        return main_transform

    def build_test_transform(self,config,mean_std):
        if config is None:
            raise ValueError("train augmentation config cannot be None.")
        required_keys = ["image"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key '{key}' in augmentation_config.")
        image = make_augmentations(config.get("image", []))
        image.append(A.Normalize(mean=mean_std["mean"], std=mean_std["std"], max_pixel_value=255.0))
        main_transform = A.Compose(
                image,  
                additional_targets={
                    "mask": "mask",
                },
            )
        return main_transform

    def build_datasets(self, only_test: bool = False) -> tuple[any, any, any]:
        root_dict = self.dataset_cropper.crop_all()
        mean_std = self.imgstats.compute_mean_std(root_dir = f"{self.root}/train/image" , output_dir=f"{self.root}/TrainsStats.yaml" )
        train_pipeline = self.build_train_transform(self.data_aug.get("train"),mean_std)
        valid_pipeline = self.build_valid_transform(self.data_aug.get("val"),mean_std)
        test_pipeline = self.build_test_transform(self.data_aug.get("test"),mean_std)
        if not only_test:
            train_dataset = WhumixDataset(image_root=root_dict["train"]["A"], mask_root=root_dict["train"]["label"], transform_pipelines=train_pipeline)
            val_dataset = WhumixDataset(image_root=root_dict["train"]["A"], mask_root=root_dict["train"]["label"], transform_pipelines=valid_pipeline)
        
        else:
            train_dataset = None
            val_dataset = None
        test_dataset = WhumixDataset(image_root=root_dict["train"]["A"], mask_root=root_dict["train"]["label"], transform_pipelines=test_pipeline)
        train_dataset = ConcatDataset([train_dataset, val_dataset])
        return train_dataset, val_dataset, test_dataset