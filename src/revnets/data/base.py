from dataclasses import dataclass, field
from typing import Any, TypeVar

import torch
from pytorch_lightning import LightningDataModule
from torch.utils import data
from torch.utils.data import DataLoader, Subset, random_split

from ..context import context

T = TypeVar("T")


@dataclass
class DataModule(LightningDataModule):
    batch_size: int = context.config.target_network_training.batch_size
    evaluation_batch_size: int = 1000
    validation_ratio = context.config.validation_ratio
    train: data.Dataset[Any] = field(init=False)
    validation: data.Dataset[Any] = field(init=False)
    train_validation: data.Dataset[Any] = field(init=False)
    test: data.Dataset[Any] = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()

    def split_train_validation(self) -> None:
        self.train, self.validation = self.split(self.train_validation)

    def split(self, dataset: data.Dataset[Any]) -> list[Subset[Any]]:
        split_sizes = [1 - self.validation_ratio, self.validation_ratio]
        seed = context.config.experiment.target_network_seed
        random_generator = torch.Generator().manual_seed(seed)
        return random_split(dataset, split_sizes, random_generator)

    def train_dataloader(self, shuffle: bool = True) -> DataLoader[Any]:
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self, shuffle: bool = False) -> DataLoader[Any]:
        return DataLoader(
            self.validation, batch_size=self.evaluation_batch_size, shuffle=False
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.test, batch_size=self.evaluation_batch_size, shuffle=False
        )
