from dataclasses import dataclass, field
from typing import cast

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from revnets.context import context
from revnets.data import base
from revnets.networks import NetworkFactory
from revnets.pipelines import Pipeline
from revnets.training import Trainer


class PredictNetwork(LightningModule):
    def __init__(self, network: torch.nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        outputs = self.network(batch[0])
        return cast(torch.Tensor, outputs)


class QueryDataSet(TensorDataset):
    def __init__(
        self,
        target: nn.Module,
        evaluation_batch_size: int = context.config.evaluation_batch_size,
    ) -> None:
        self.target = target
        self.evaluation_batch_size = evaluation_batch_size
        tensors = torch.Tensor([]), torch.Tensor([])
        super().__init__(*tensors)

    def add(self, queries: torch.Tensor) -> None:
        targets = self.compute_targets(queries)
        self.tensors = (
            torch.cat([self.tensors[0], queries], dim=0),
            torch.cat([self.tensors[1], targets], dim=0),
        )

    def compute_targets(self, inputs: torch.Tensor) -> torch.Tensor:
        dataset = TensorDataset(inputs)
        dataloader = DataLoader(dataset, batch_size=self.evaluation_batch_size)
        trainer = Trainer(barebones=True)
        network = PredictNetwork(self.target)
        untyped_batch_outputs = trainer.predict(network, dataloader)
        batch_outputs = cast(list[torch.Tensor], untyped_batch_outputs)
        return torch.cat(batch_outputs)


@dataclass
class DataModule(base.DataModule):
    pipeline: Pipeline = field(default_factory=lambda: Pipeline(NetworkFactory()))
    batch_size: int = context.config.reconstruction_training.batch_size
    evaluation_batch_size = context.config.evaluation_batch_size
    train: QueryDataSet = field(init=False)
    validation: QueryDataSet = field(init=False)
    test: QueryDataSet = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        target = self.pipeline.create_target_network()
        self.train = QueryDataSet(target, self.evaluation_batch_size)
        self.validation = QueryDataSet(target, self.evaluation_batch_size)
        self.test = QueryDataSet(target, self.evaluation_batch_size)
