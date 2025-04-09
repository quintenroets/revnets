from typing import cast

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from revnets.context import context
from revnets.training import Trainer


class PredictNetwork(LightningModule):
    def __init__(self, network: torch.nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        outputs = self.network(batch[0])
        return cast("torch.Tensor", outputs)


def compute_targets(
    inputs: torch.Tensor,
    network: nn.Module,
    batch_size: int | None = None,
) -> torch.Tensor:
    if batch_size is None:
        batch_size = context.config.evaluation_batch_size
    dataset = TensorDataset(inputs)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    trainer = Trainer(barebones=True)
    network = PredictNetwork(network)
    untyped_batch_outputs = trainer.predict(network, dataloader)
    batch_outputs = cast("list[torch.Tensor]", untyped_batch_outputs)
    return torch.cat(batch_outputs)
