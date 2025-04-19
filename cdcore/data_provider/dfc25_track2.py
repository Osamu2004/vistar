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
from os.path import splitext
from os import listdir
from cdcore.data_provider.base import BitemporalDataset
from apps.builder import make_augmentations
from apps.registry import DATAPROVIDER
from apps.cropper import DatasetCropper,ImgStats
from torch.utils.data import Dataset

class Dfc25_track2_Dataset(Dataset):
    def __init__(self, 
                 t1_image_root, 
                 t2_image_root,
                 change_mask_root=None,
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

        self.check_ids()

    def __len__(self):
        return len(self.t1_ids)


    def __getitem__(self, idx):
        # 获取文件 ID
        name = self.t1_ids[idx]

        # 构造文件路径
        image1_path = f"{self.t1_image_root}/{name}{self.imagetype}"
        image2_path = f"{self.t2_image_root}/{name.replace('_pre_disaster', '_post_disaster')}{self.imagetype}"

        # 加载 T1 和 T2 图像
        image1 = io.imread(image1_path)
        image2 = io.imread(image2_path)

        # 加载变更掩码（如果提供）

        if self.change_mask_root:
            label_path = f"{self.change_mask_root}/{name.replace('_pre_disaster', '_building_damage')}{self.imagetype}"
            changelabel = io.imread(label_path)
        else:
            changelabel=None
        sample = {
            "image": image1,
            "t2_image": image2,
        }
        if changelabel is not None:
            sample["mask"] = changelabel
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
        sample["name"] = f"{name.replace('_pre_disaster', '_building_damage')}"
        return sample

    def check_ids(self):
        # 动态计算 T2 和其他 ID 列表
        t1_ids = sorted([splitext(file.replace('_pre_disaster', ''))[0] for file in listdir(self.t1_image_root) if not file.startswith('.')])
        t2_ids = sorted([splitext(file.replace('_post_disaster', ''))[0] for file in listdir(self.t2_image_root) if not file.startswith('.')])
        if self.change_mask_root:
            change_ids = sorted([splitext(file.replace('_building_damage', ''))[0] for file in listdir(self.change_mask_root) if not file.startswith('.')])
        else:
            change_ids = None
        



        # 检查 T1 和 T2 图像 ID
        assert len(t1_ids) == len(t2_ids), "T1 and T2 IDs have different lengths."
        for t1, t2 in zip(t1_ids, t2_ids):
            assert t1 == t2, f"Mismatch in T1 and T2 IDs: {t1} vs {t2}"

        # 检查变更掩码 ID（如果提供）
        if change_ids:
            assert len(t1_ids) == len(change_ids), "T1 and change IDs have different lengths."
            for t1, change in zip(t1_ids, change_ids):
                assert t1 == change, f"Mismatch in T1 and change IDs: {t1} vs {change}"


@DATAPROVIDER.register("dfc25_track2")
class Dfc25_track2_DataProvider(DataProvider):
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
        self.main_classes = 4
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
                },
            )
        return main_transform, t1_transform, t2_transform

    def build_datasets(self, only_test: bool = False) -> tuple[any, any, any]:
        root_dict = self.dataset_cropper.crop_all()
        t1_mean_std = self.imgstats.compute_mean_std(root_dir=f"{self.root}/train/pre-event", output_dir=f"{self.root}/t1TrainsStats.yaml")
        t2_mean_std = self.imgstats.compute_mean_std(root_dir=f"{self.root}/train/post-event", output_dir=f"{self.root}/t2TrainsStats.yaml")
        
        train_pipeline = self.build_train_transform(self.data_aug.get("train"), t1_mean_std, t2_mean_std)
        valid_pipeline = self.build_valid_transform(self.data_aug.get("val"), t1_mean_std, t2_mean_std)
        val_dataset = None
        # 仅在不是测试模式时，加载训练集和验证集
        if not only_test:
            train_dataset = Dfc25_track2_Dataset(
                    t1_image_root=root_dict["train"]["pre-event"], 
                    t2_image_root=root_dict["train"]["post-event"], 
                    change_mask_root=root_dict["train"]["target"], 
                    transform_pipelines=train_pipeline
                )
        else:
            train_dataset = None
        test_dataset = Dfc25_track2_Dataset(
            t1_image_root=root_dict["val"]["pre-event"], 
            t2_image_root=root_dict["val"]["post-event"], 
            transform_pipelines=valid_pipeline
        )
        # 返回数据集
        return train_dataset, val_dataset, test_dataset