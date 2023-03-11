from dataclasses import dataclass

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..utils import Trainer
from . import base
from .base import Split


class PredictModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model(batch[0])


@dataclass
class Dataset(base.Dataset):
    original_dataset: base.Dataset
    target_network: torch.nn.Module

    def __post_init__(self):
        super().__init__()
        self.evaluator_model = PredictModel(self.target_network)

    def prepare_data(self, stage: str = None) -> None:
        self.original_dataset.prepare_data()
        self.original_dataset.calibrate(self.evaluator_model)
        self.train_val_dataset = self.generate_dataset(Split.train_val)
        self.test_dataset = self.generate_dataset(Split.test)

    def generate_dataset(self, split: Split):
        if self.original_dataset.get_dataset(split) is not None:
            return self.add_output_supervision(split)

    def add_output_supervision(self, split: Split):
        dataloader = self.original_dataset.get_dataloader(
            split, self.original_dataset.eval_batch_size
        )
        inputs = self.get_inputs(dataloader)
        targets = self.get_output_targets(dataloader)
        return TensorDataset(inputs, targets)

    @classmethod
    def get_inputs(cls, dataloader: DataLoader):
        input_batches = tuple(batch[0] for batch in dataloader)
        return torch.cat(input_batches)

    def get_output_targets(self, dataloader: DataLoader):
        model = PredictModel(self.target_network)
        trainer = Trainer()
        batch_outputs = trainer.predict(model, dataloader)
        return torch.cat(batch_outputs)
