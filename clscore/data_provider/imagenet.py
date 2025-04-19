import copy
import math
import os
from typing import Any, Optional
import albumentations as A
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from apps.builder import make_augmentations
from apps.registry import DATAPROVIDER
from apps.cropper import DatasetCropper,ImgStats
from apps.data_provider import DataProvider
from albumentations.pytorch import ToTensorV2
__all__ = ["ImageNetDataProvider"]


class ImageNetDataProvider(DataProvider):
    name = "imagenet"

    n_classes = 1000

    def __init__(
        self,
        root: str,
        data_aug: Optional[dict | list[dict]] = None,
        ###########################################
        train_batch_size=128,
        test_batch_size=128,
        valid_size: Optional[int | float] = None,
        n_worker=8,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        train_ratio: Optional[float] = None,
        drop_last: bool = False,
        only_test: bool = False,
    ):
        self.root = root
        self.data_aug = data_aug
        self.main_classes = 1000
        self.samples = [
            "image",  
            "label",
        ]
        self.imgstats = ImgStats()
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

    def build_train_transform(self,config,mean_std):
        if config is None:
            raise ValueError("train augmentation config cannot be None.")
        required_keys = ["image"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key '{key}' in augmentation_config.")
        image = make_augmentations(config.get("image", []))
        image.append(A.Normalize(mean=mean_std["mean"], std=mean_std["std"], max_pixel_value=255.0))
        image.append(ToTensorV2())
        
        main_transform = A.Compose(
                image,  
            )
        return main_transform

    def build_valid_transform(self,config,mean_std):
        if config is None:
            raise ValueError("val augmentation config cannot be None.")
        required_keys = ["image"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key '{key}' in augmentation_config.")
        image = make_augmentations(config.get("image", []))
        image.append(A.Normalize(mean=mean_std["mean"], std=mean_std["std"], max_pixel_value=255.0))
        image.append(ToTensorV2())
        
        main_transform = A.Compose(
                image,  
            )
        return main_transform


    def build_datasets(self ,only_test: bool = False) -> tuple[Any, Any, Any]:
        #mean_std = self.imgstats.compute_mean_std(root_dir = f"{self.root}/mmbuilding/rgb" , output_dir=f"{self.root}/TrainsStats.yaml" )
        mean_std = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        train_pipeline = self.build_train_transform(self.data_aug.get("train"),mean_std)
        valid_pipeline = self.build_valid_transform(self.data_aug.get("val"),mean_std)

        train_dataset = ImageFolder(os.path.join(self.data_dir, "train"), train_pipeline )
        test_dataset = ImageFolder(os.path.join(self.data_dir, "val"), valid_pipeline)
        val_dataset =None
        return train_dataset, val_dataset, test_dataset