from dataclasses import dataclass, field
from typing import Any, TypeVar

import torch
from pytorch_lightning import LightningDataModule
from torch.utils import data
from torch.utils.data import DataLoader, random_split

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
        split_sizes = self.calculate_split_sizes()
        seed = context.config.experiment.target_network_seed
        random_generator = torch.Generator().manual_seed(seed)
        split = random_split(self.train_validation, split_sizes, random_generator)
        self.train, self.validation = split

    def calculate_split_sizes(self) -> tuple[int, int]:
        total_size = len(self.train_validation)  # type: ignore[arg-type]
        validation_size = int(self.validation_ratio * total_size)
        train_size = total_size - validation_size
        return train_size, validation_size

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
