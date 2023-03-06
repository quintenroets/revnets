from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset

from ...data import mnist1d
from ...data.split import Split
from ...utils.trainer import Trainer
from . import base
from .base import ReconstructModel


@dataclass
class Dataset(mnist1d.Dataset):
    train_datasets: list[data.Dataset] = field(default_factory=list)
    val_datasets: list[data.Dataset] = field(default_factory=list)

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
    samples_per_stage: int = 1000
    validation_ratio: float = 0.2

    @property
    def val_samples_per_stage(self):
        return int(self.samples_per_stage * self.validation_ratio)

    def __post_init__(self):
        self.trainer = Trainer()
        self.data = Dataset()

    def start_training(self):
        model = ReconstructModel(self.original, self.reconstruction)
        self.extend_dataset()
        self.data.calibrate(model)
        self.trainer.fit(model, self.data)

    def extend_dataset(self):
        new_dataset = self.get_new_dataset(self.samples_per_stage)
        self.data.add_dataset(new_dataset, Split.train)
        new_dataset = self.get_new_dataset(self.samples_per_stage)
        self.data.add_dataset(new_dataset, Split.valid)

    def get_new_dataset(self, number_of_samples):
        input_shape = (40,)
        batch_input_shape = number_of_samples, *input_shape
        new_inputs = torch.rand(batch_input_shape)
        new_outputs = self.get_outputs(new_inputs, self.original)
        dataset = TensorDataset(new_inputs, new_outputs)
        return dataset

    def get_outputs(self, inputs: torch.Tensor, model: torch.nn.Module):
        dataset = TensorDataset(inputs)
        dataloader = DataLoader(dataset, batch_size=self.data.batch_size * 2)
        predict_model = PredictModel(model)
        new_output_batches = self.trainer.predict(predict_model, dataloaders=dataloader)
        new_outputs = torch.cat(new_output_batches)
        return new_outputs
