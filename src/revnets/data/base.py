import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from torch.utils import data
from torch.utils.data import ConcatDataset, Subset

from .. import utils
from ..context import context
from .split import Split
from .utils import split_train_val


class Dataset(pl.LightningDataModule):
    def __init__(
        self,
        repetition_factor: float | None = None,
        validation_ratio: float | None = None,
    ) -> None:
        super().__init__()
        self.train_val_dataset: data.Dataset | None = None
        self.train_dataset: data.Dataset | None = None
        self.val_dataset: data.Dataset | None = None
        self.test_dataset: data.Dataset | None = None
        self.batch_size: int = context.config.target_network_training.batch_size
        self.eval_batch_size: int | None = None
        self.repetition_factor: float | None = repetition_factor
        self.validation_ratio: float | None = validation_ratio
        """
        For data size experiments, we want to now how many samples we need in order to
        have fair comparisons, we keep the number of effective samples the same by
        scaling the number of repetitions in the training set.
        """

    def prepare(self) -> None:
        self.prepare_data()
        self.setup("train")

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = None) -> None:
        # allow overriding train_dataset
        if self.train_dataset is None:
            self.train_dataset, self.val_dataset = split_train_val(
                self.train_val_dataset, val_fraction=self.validation_ratio
            )

    def train_dataloader(self, shuffle: bool = True, batch_size: int | None = None):
        if batch_size is None:
            batch_size = self.batch_size
        return self.get_dataloader(
            Split.train, batch_size, shuffle=shuffle, use_repeat=True
        )

    def val_dataloader(self, shuffle: bool = False):
        return self.get_dataloader(Split.valid, self.eval_batch_size, shuffle=shuffle)

    def test_dataloader(self, shuffle: bool = False):
        return self.get_dataloader(Split.test, self.eval_batch_size, shuffle=shuffle)

    def get_all_inputs(self, split: Split):
        dataloader = self.get_dataloader(
            split, batch_size=self.eval_batch_size, shuffle=False
        )
        batched_inputs = tuple(batch[0] for batch in dataloader)
        return torch.vstack(batched_inputs)

    def get_all_targets(self, split: Split):
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
    ):
        if context.config.debug:
            # memorize 1 training batch during debugging
            dataset = self.get_dataset(Split.train)
            dataset = Subset(dataset, list(range(self.batch_size)))
            if split.is_train:
                dataset = ConcatDataset([dataset for _ in range(100)])
            shuffle = False
        else:
            dataset = self.get_dataset(split)
            if split.is_train and self.repetition_factor is not None and use_repeat:
                repetition_factor_int = int(self.repetition_factor)
                repetition_fraction = self.repetition_factor - repetition_factor_int
                datasets = [dataset] * repetition_factor_int
                if repetition_fraction:
                    last_length = int(len(dataset) * repetition_fraction)  # noqa
                    last_dataset = Subset(dataset, list(range(last_length)))
                    datasets.append(last_dataset)

                dataset = ConcatDataset(datasets)

        if batch_size == -1:
            batch_size = len(dataset)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            # num_workers=config.num_workers,
            shuffle=shuffle,
            # persistent_workers=True,
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

    def calibrate(self, network: LightningModule) -> None:
        self.eval_batch_size = utils.batch_size.get_max_batch_size(model, self)
