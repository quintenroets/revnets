from dataclasses import dataclass, field

import torch
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils import data
from torch.utils.data import ConcatDataset, DataLoader, Subset

from revnets import training
from revnets.training import Network

from ..context import context
from .split import Split
from .utils import split_train_val


@dataclass
class Dataset(LightningDataModule):
    """
    For data size experiments, we want to now how many samples we need in order to have
    fair comparisons, we keep the number of effective samples the same by scaling the
    number of repetitions in the training set.
    """

    eval_batch_size: int = field(init=False)
    train_val_dataset: data.Dataset[tuple[torch.Tensor, ...]] = field(init=False)
    train_dataset: data.Dataset[tuple[torch.Tensor, ...]] = field(init=False)
    val_dataset: data.Dataset[tuple[torch.Tensor, ...]] = field(init=False)
    test_dataset: data.Dataset[tuple[torch.Tensor, ...]] = field(init=False)
    repetition_factor: float | None = None
    batch_size: int = context.config.target_network_training.batch_size
    validation_ratio = context.config.validation_ratio

    def prepare(self) -> None:
        self.prepare_data()
        self.setup("train")

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str | None = None) -> None:
        # allow overriding train_dataset
        if self.train_dataset is None:
            assert self.train_val_dataset is not None
            self.train_dataset, self.val_dataset = split_train_val(
                self.train_val_dataset, val_fraction=self.validation_ratio
            )

    def train_dataloader(
        self, shuffle: bool = True, batch_size: int | None = None
    ) -> DataLoader[tuple[torch.Tensor, ...]]:
        if batch_size is None:
            batch_size = self.batch_size
        return self.get_dataloader(
            Split.train, batch_size, shuffle=shuffle, use_repeat=True
        )

    def val_dataloader(
        self, shuffle: bool = False
    ) -> DataLoader[tuple[torch.Tensor, ...]]:
        assert self.eval_batch_size is not None
        return self.get_dataloader(Split.valid, self.eval_batch_size, shuffle=shuffle)

    def test_dataloader(
        self, shuffle: bool = False
    ) -> DataLoader[tuple[torch.Tensor, ...]]:
        assert self.eval_batch_size is not None
        return self.get_dataloader(Split.test, self.eval_batch_size, shuffle=shuffle)

    def get_all_inputs(self, split: Split) -> torch.Tensor:
        assert self.eval_batch_size is not None
        dataloader = self.get_dataloader(
            split, batch_size=self.eval_batch_size, shuffle=False
        )
        batched_inputs = tuple(batch[0] for batch in dataloader)
        return torch.vstack(batched_inputs)

    def get_all_targets(self, split: Split) -> torch.Tensor:
        assert self.eval_batch_size is not None
        dataloader = self.get_dataloader(
            split, batch_size=self.eval_batch_size, shuffle=False
        )
        batched_targets = tuple(batch[1] for batch in dataloader)
        return torch.vstack(batched_targets)

    def get_dataloader(
        self,
        split: Split,
        batch_size: int,
        shuffle: bool = False,
        use_repeat: bool = False,
    ) -> DataLoader[tuple[torch.Tensor, ...]]:
        dataset = (
            self.create_debug_dataset(split)
            if context.config.debug
            else self.create_dataset(split, use_repeat)
        )
        if batch_size == -1:
            batch_size = len(dataset)  # type: ignore[arg-type]
        if context.config.debug:
            shuffle = False
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def create_debug_dataset(
        self, split: Split
    ) -> data.Dataset[tuple[torch.Tensor, ...]]:
        # memorize 1 training batch during debugging
        dataset = self.get_dataset(Split.train)
        dataset = Subset(dataset, list(range(self.batch_size)))
        if split.is_train:
            dataset = ConcatDataset([dataset for _ in range(100)])
        return dataset

    def create_dataset(
        self, split: Split, use_repeat: bool
    ) -> data.Dataset[tuple[torch.Tensor, ...]]:
        dataset = self.get_dataset(split)
        if split.is_train and self.repetition_factor is not None and use_repeat:
            repetition_factor_int = int(self.repetition_factor)
            repetition_fraction = self.repetition_factor - repetition_factor_int
            datasets = [dataset] * repetition_factor_int
            if repetition_fraction:
                last_length = int(len(dataset) * repetition_fraction)  # type: ignore[arg-type]
                last_dataset = Subset(dataset, list(range(last_length)))
                datasets.append(last_dataset)

            dataset = ConcatDataset(datasets)
        return dataset

    def get_dataset(self, datatype: Split) -> data.Dataset[tuple[torch.Tensor, ...]]:
        match datatype:
            case Split.train:
                dataset = self.train_dataset
            case Split.valid:
                dataset = self.val_dataset
            case Split.test:
                dataset = self.test_dataset
            case Split.train_val:
                dataset = self.train_val_dataset
        return dataset

    def calibrate(self, network: Network | LightningModule) -> None:
        self.eval_batch_size = training.calculate_max_batch_size(network, self)
