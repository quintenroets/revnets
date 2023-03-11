import pytorch_lightning as pl
import torch
from torch.utils import data
from torch.utils.data import ConcatDataset, Subset

from .. import utils
from ..utils import config
from .split import Split
from .utils import split_train_val


class Dataset(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_val_dataset: data.Dataset | None = None
        self.train_dataset: data.Dataset | None = None
        self.val_dataset: data.Dataset | None = None
        self.test_dataset: data.Dataset | None = None
        self.batch_size: int = self.calculate_effective_batch_size(config.batch_size)
        self.eval_batch_size: int | None = None

    @classmethod
    def calculate_effective_batch_size(cls, batch_size):
        """
        We want reproducible experiments.

        => effective batch size needs to remain the same at all times
           effective batch size = batch size per gpu * # GPUs
        => we can only increase the number of GPUs when we can scale down
           the batch size per gpu accordingly
        """
        used_devices = 1
        batch_size_per_gpu = batch_size
        while 2 * used_devices <= config.num_devices and batch_size_per_gpu % 2 == 0:
            batch_size_per_gpu //= 2
            used_devices *= 2

        config._num_devices = used_devices
        return batch_size_per_gpu

    def prepare(self):
        self.prepare_data()
        self.setup("train")

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = None) -> None:
        self.train_dataset, self.val_dataset = split_train_val(self.train_val_dataset)

    def train_dataloader(self, shuffle=True):
        return self.get_dataloader(Split.train, self.batch_size, shuffle=shuffle)

    def val_dataloader(self, shuffle=False):
        return self.get_dataloader(Split.valid, self.eval_batch_size, shuffle=shuffle)

    def test_dataloader(self, shuffle=False):
        return self.get_dataloader(Split.test, self.eval_batch_size, shuffle=shuffle)

    def get_dataloader(self, split: Split, batch_size, shuffle=False):
        if config.debug:
            # memorize 1 training batch during debugging
            dataset = self.get_dataset(Split.train)
            dataset = Subset(dataset, list(range(self.batch_size)))
            if split.is_train:
                dataset = ConcatDataset([dataset for _ in range(100)])
            shuffle = False
        else:
            dataset = self.get_dataset(split)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            persistent_workers=True,
        )
        return dataloader

    def get_dataset(self, datatype):
        match datatype:
            case Split.train:
                dataset = self.train_dataset
            case Split.valid:
                dataset = self.val_dataset
            case Split.test:
                dataset = self.test_dataset
            case Split.train_val:
                dataset = self.train_val_dataset
        return dataset  # noqa

    def calibrate(self, model):
        self.eval_batch_size = utils.batch_size.get_max_batch_size(model, self)
