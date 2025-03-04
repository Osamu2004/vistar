

import copy
import warnings

import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
from typing import Optional

__all__ = [ "DataProvider"]




def random_drop_data(dataset, drop_size: int, seed: int, keys=("samples",)):
    g = torch.Generator()
    g.manual_seed(seed)  # set random seed before sampling validation set
    rand_indexes = torch.randperm(len(dataset), generator=g).tolist()

    dropped_indexes = rand_indexes[:drop_size]
    remaining_indexes = rand_indexes[drop_size:]

    dropped_dataset = copy.deepcopy(dataset)
    for key in keys:
        setattr(dropped_dataset, key, [getattr(dropped_dataset, key)[idx] for idx in dropped_indexes])
        setattr(dataset, key, [getattr(dataset, key)[idx] for idx in remaining_indexes])
    return dataset, dropped_dataset


class DataProvider:
    data_keys = ("samples",)
    SUB_SEED = 937162211  # random seed for sampling subset
    VALID_SEED = 2147483647  # random seed for the validation set

    name: str

    def __init__(
        self,
        n_worker: int,
        train_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        train_ratio: Optional[float] = None,
        drop_last: bool = False,
        only_test: bool = False,  # Only test mode flag
    ):
        warnings.filterwarnings("ignore")
        super().__init__()

        # batch_size & valid_size
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size or self.train_batch_size

        # distributed configs
        self.num_replicas = num_replicas
        self.rank = rank

        # build datasets (only load test set if only_test is True)
        train_dataset, val_dataset, test_dataset = self.build_datasets(only_test)
        print(train_dataset,val_dataset,test_dataset,123123)

        if train_ratio is not None and train_ratio < 1.0 and not only_test:
            assert 0 < train_ratio < 1
            _, train_dataset = random_drop_data(
                train_dataset,
                int(train_ratio * len(train_dataset)),
                self.SUB_SEED,
                self.data_keys,
            )
        print(only_test)
        # build data loader (load only test dataset if only_test is True)
        if not only_test:
            self.train = self.build_dataloader(train_dataset, train_batch_size, n_worker, drop_last=True, train=True)
            print(self.train)
            self.valid = self.build_dataloader(val_dataset, test_batch_size, n_worker, drop_last=False, train=False)
            self.test = self.build_dataloader(test_dataset, test_batch_size, n_worker, drop_last=False, train=False)
            if self.valid is None:
                self.valid = self.test
        else:
            # In test-only mode, do not load train and valid loaders
            self.train = None
            self.valid = None
            self.test = self.build_dataloader(test_dataset, test_batch_size, n_worker, drop_last=False, train=False)
            print(1)

    @property
    def data_shape(self) -> tuple[int, ...]:
        return 3, self.active_image_size[0], self.active_image_size[1]

    def build_valid_transform(self, config: dict) -> any:
        raise NotImplementedError

    def build_train_transform(self, config: dict) -> any:
        raise NotImplementedError
    
    def build_test_transform(self, config: dict) -> any:
        raise NotImplementedError

    def build_datasets(self, only_test: bool) -> tuple[any, any, any]:
        raise NotImplementedError

    def build_dataloader(self, dataset: any or None, batch_size: int, n_worker: int, drop_last: bool, train: bool):
        print(dataset,"build_dataloader")
        if dataset is None:
            print(123123)
            return None
        else:
            dataloader_class = torch.utils.data.DataLoader

        if train:

            sampler = InfiniteSampler(dataset, shuffle=True)
            return dataloader_class(
                dataset=dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=n_worker,
                pin_memory=True,
                drop_last=drop_last,
            )
                
        else:
            return dataloader_class(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=n_worker,
                pin_memory=True,
                drop_last=drop_last,
            )

    def set_epoch(self, epoch: int) -> None:
        if isinstance(self.train.sampler, DistributedSampler):
            self.train.sampler.set_epoch(epoch)


from torch.utils.data import Sampler
import random
class InfiniteSampler(Sampler):
    """A sampler that endlessly loops over the dataset."""
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.length = len(dataset)

    def __iter__(self):
        while True:
            indices = list(range(self.length))
            if self.shuffle:
                random.shuffle(indices)
            for idx in indices:
                yield idx

    def __len__(self):
        # Infinite loop, so return a large number
        return float('inf')
'''
class InfiniteSampler(Sampler):
    """It's designed for iteration-based runner and yields a mini-batch indices
    each time.

    The implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/distributed_sampler.py

    Args:
        dataset (Sized): The dataset.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed. If None, set a random seed.
            Defaults to None.
    """  # noqa: W605

    def __init__(self,
                 dataset,
                 shuffle: bool = True,
                 seed = None) -> None:

        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.size = len(dataset)

    def _infinite_indices(self):
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(self.size, generator=g).tolist()

            else:
                yield from torch.arange(self.size).tolist()


    def __iter__(self):
        """Iterate the indices."""
        yield from self._infinite_indices()

    def __len__(self) -> int:
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch: int) -> None:
        """Not supported in iteration-based runner."""
        pass

'''
import bisect
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader
class _InfiniteDataloaderIterator:
    """An infinite dataloader iterator wrapper for IterBasedTrainLoop.

    It resets the dataloader to continue iterating when the iterator has
    iterated over all the data. However, this approach is not efficient, as the
    workers need to be restarted every time the dataloader is reset. It is
    recommended to use `mmengine.dataset.InfiniteSampler` to enable the
    dataloader to iterate infinitely.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self._dataloader = dataloader
        self._iterator = iter(self._dataloader)
        self._epoch = 0

    def __iter__(self):
        return self

    def __next__(self) -> Sequence[dict]:
        try:
            data = next(self._iterator)
        except StopIteration:
            print(
                'Reach the end of the dataloader, it will be '
                'restarted and continue to iterate. It is '
                'recommended to use '
                '`mmengine.dataset.InfiniteSampler` to enable the '
                'dataloader to iterate infinitely.'
            )
            self._epoch += 1
            if hasattr(self._dataloader, 'sampler') and hasattr(
                    self._dataloader.sampler, 'set_epoch'):
                # In case the` _SingleProcessDataLoaderIter` has no sampler,
                # or data loader uses `SequentialSampler` in Pytorch.
                self._dataloader.sampler.set_epoch(self._epoch)

            elif hasattr(self._dataloader, 'batch_sampler') and hasattr(
                    self._dataloader.batch_sampler.sampler, 'set_epoch'):
                # In case the` _SingleProcessDataLoaderIter` has no batch
                # sampler. batch sampler in pytorch warps the sampler as its
                # attributes.
                self._dataloader.batch_sampler.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self._iterator = iter(self._dataloader)
            data = next(self._iterator)
        return data