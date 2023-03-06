from dataclasses import dataclass, field

import cli
import pytorch_lightning as pl
import torch
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset

from ...data import mnist1d
from ...data.split import Split
from ...utils import config
from ...utils.trainer import Trainer
from . import base
from .base import ReconstructModel


@dataclass
class Dataset(mnist1d.Dataset):
    train_datasets: list[data.Dataset] = field(default_factory=list)
    val_datasets: list[data.Dataset] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        batch_size = config.reconstruction_batch_size
        self.batch_size = self.calculate_effective_batch_size(batch_size)

    def setup(self, stage: str = None) -> None:
        pass

    def add_dataset(self, dataset: data.Dataset, split: Split):
        match split:
            case Split.train:
                self.train_datasets.append(dataset)
                self.train_dataset = torch.utils.data.ConcatDataset(self.train_datasets)
            case Split.valid:
                self.val_datasets.append(dataset)
                self.val_dataset = torch.utils.data.ConcatDataset(self.val_datasets)


class PredictModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        inputs = batch[0]
        return self.model(inputs)


@dataclass
class Reconstructor(base.Reconstructor):
    samples_per_stage: int = 100000
    validation_ratio: float = 0.2
    num_extensions: int = 1
    always_train: bool = True

    @property
    def val_samples_per_stage(self):
        return int(self.samples_per_stage * self.validation_ratio)

    def __post_init__(self):
        self.trainer = Trainer()
        self.data = Dataset()
        self.model = ReconstructModel(self.original, self.reconstruction)

    def start_training(self):
        model = ReconstructModel(self.original, self.reconstruction)
        for extension_iteration in range(self.num_extensions):
            cli.console.rule(str(extension_iteration))
            self.extend_dataset()
            trainer = Trainer()
            trainer.fit(model, self.data)

    def extend_dataset(self):
        first_extension = len(self.data.train_datasets) == 0
        new_dataset = self.get_new_dataset(self.samples_per_stage)
        self.data.add_dataset(new_dataset, Split.train)
        new_dataset = self.get_new_dataset(self.samples_per_stage)
        self.data.add_dataset(new_dataset, Split.valid)
        if first_extension:
            self.data.calibrate(self.model)

    def get_new_dataset(self, number_of_samples):
        input_shape = (40,)
        batch_input_shape = number_of_samples, *input_shape
        new_inputs = torch.randn(batch_input_shape)
        # same mean and std as training data
        new_outputs = self.get_outputs(new_inputs, self.original)
        dataset = TensorDataset(new_inputs, new_outputs)
        return dataset

    def get_outputs(self, inputs: torch.Tensor, model: torch.nn.Module):
        dataset = TensorDataset(inputs)
        dataloader = DataLoader(
            dataset, batch_size=self.data.batch_size, num_workers=config.num_workers
        )
        predict_model = PredictModel(model)
        new_output_batches = self.trainer.predict(predict_model, dataloaders=dataloader)
        new_outputs = torch.cat(new_output_batches)
        return new_outputs
