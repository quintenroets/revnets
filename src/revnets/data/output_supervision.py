from dataclasses import dataclass, field
from typing import Any, cast

import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, TensorDataset

from revnets.training import Trainer

from . import base
from .base import Split


class PredictNetwork(LightningModule):
    def __init__(self, network: torch.nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        outputs = self.network(batch[0])
        return cast(torch.Tensor, outputs)

    def get_predictions(
        self, trainer: Trainer, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        batch_outputs = trainer.predict(self, *args, **kwargs)
        outputs = (
            torch.Tensor()
            if batch_outputs is None
            else torch.cat(cast(list[torch.Tensor], batch_outputs))
        )
        return outputs


@dataclass
class Dataset(base.Dataset):
    original_dataset: base.Dataset = field(default_factory=base.Dataset)
    target_network: torch.nn.Module = field(default_factory=torch.nn.Module)

    def __post_init__(self) -> None:
        self.evaluator_model = PredictNetwork(self.target_network)

    def prepare_data(self, stage: str | None = None) -> None:
        self.original_dataset.prepare_data()
        self.original_dataset.calibrate(self.evaluator_model)
        self.train_val_dataset = self.generate_dataset(Split.train_val)
        self.test_dataset = self.generate_dataset(Split.test)

    def generate_dataset(self, split: Split) -> TensorDataset:
        assert self.original_dataset.get_dataset(split) is not None
        return self.add_output_supervision(split)

    def add_output_supervision(self, split: Split) -> TensorDataset:
        assert self.original_dataset.eval_batch_size is not None
        dataloader = self.original_dataset.get_dataloader(
            split, self.original_dataset.eval_batch_size
        )
        inputs = self.get_inputs(dataloader)
        targets = self.get_output_targets(dataloader)
        return TensorDataset(inputs, targets)

    @classmethod
    def get_inputs(
        cls, dataloader: DataLoader[tuple[torch.Tensor, ...]]
    ) -> torch.Tensor:
        input_batches = tuple(batch[0] for batch in dataloader)
        return torch.cat(input_batches)

    def get_output_targets(
        self, dataloader: DataLoader[tuple[torch.Tensor, ...]]
    ) -> torch.Tensor:
        return self.get_predictions(dataloader, self.target_network)

    @classmethod
    def get_predictions(
        cls, dataloader: DataLoader[tuple[torch.Tensor, ...]], network: torch.nn.Module
    ) -> torch.Tensor:
        predict_model = PredictNetwork(network)
        trainer = Trainer(barebones=True)
        return predict_model.get_predictions(trainer, dataloader)
