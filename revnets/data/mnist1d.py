import pickle
from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
import requests
import torch
from torch.utils.data import ConcatDataset, Subset, TensorDataset

from ..utils import Path, batch_size, config
from .split import Split


def check_existence(path):
    if not path.exists():
        url = "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"
        path.byte_content = requests.get(url, allow_redirects=True).content


@dataclass
class Dataset(pl.LightningDataModule):
    train_dataset: torch.utils.data.Dataset = None
    val_dataset: torch.utils.data.Dataset = None
    test_dataset: torch.utils.data.Dataset = None
    eval_batch_size: int = None

    def __post_init__(self):
        super().__init__()
        self.batch_size = self.calculate_effective_batch_size()

    @classmethod
    def calculate_effective_batch_size(cls):
        """
        We want reproducible experiments.

        => effective batch size needs to remain the same at all times
           effective batch size = batch size per gpu * # GPUs
        => we can only increase the number of GPUs when we can scale down
           the batch size per gpu accordingly
        """
        used_devices = 1
        batch_size_per_gpu = config.batch_size
        while 2 * used_devices <= config.num_devices and batch_size_per_gpu % 2 == 0:
            batch_size_per_gpu //= 2
            used_devices *= 2

        config._num_devices = used_devices
        return batch_size_per_gpu

    def setup(self, stage: str = None) -> None:
        data = self.get_data()

        def load(name: str):
            return torch.from_numpy(data[name])

        x = data["x"]
        y = data["y"]
        x_test = data["x_test"]
        y_test = data["y_test"]

        def prepare_x(x_array: np.ndarray):
            x_torch = torch.Tensor(x_array)
            x_torch = torch.nn.functional.normalize(x_torch)
            return x_torch

        x = prepare_x(x)
        x_test = prepare_x(x_test)

        y = torch.LongTensor(y)
        y_test = torch.LongTensor(y_test)

        train_dataset = TensorDataset(x, y)
        self.train_dataset, self.val_dataset = split_train_val(train_dataset)
        self.test_dataset = TensorDataset(x_test, y_test)

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
        return dataset  # noqa

    @classmethod
    def get_data(cls):
        path = Path.data / "minst_1D.pkl"
        check_existence(path)
        with path.open("rb") as fp:
            return pickle.load(fp)

    def calibrate(self, model):
        self.eval_batch_size = batch_size.get_max_batch_size(model, self)


def split_train_val(
    dataset: torch.utils.data.Dataset, val_fraction=0.1
) -> tuple[Dataset, ...]:
    # ignore len(dataset) warning
    total_size = len(dataset)  # noqa
    val_size = int(val_fraction * total_size)
    train_size = total_size - val_size
    sizes = [train_size, val_size]
    # Make split deterministic for reproducibility
    split_generator = torch.Generator().manual_seed(config.manual_seed)
    train_data, val_data = torch.utils.data.random_split(
        dataset, sizes, generator=split_generator
    )
    return train_data, val_data
