import math
from dataclasses import dataclass, field
from typing import Any

import cli
import torch
from pytorch_lightning import LightningModule
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from revnets.context import context
from revnets.reconstructions.queries.base import DataModule
from revnets.training import Trainer
from revnets.training.reconstructions import Network

from . import base


class InputNetwork(LightningModule):
    def __init__(
        self,
        shape: tuple[int, ...],
        reconstructions: list[torch.nn.Sequential],
        learning_rate: float | None = None,
        *,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.shape = shape
        if learning_rate is None:
            learning_rate = context.config.difficult_inputs_training.learning_rate
        self.learning_rate = learning_rate
        self.inputs_embedding = self.create_input_embeddings(shape)
        self.reconstructions = torch.nn.ModuleList(reconstructions)
        self.verbose = verbose

    def on_train_start(self) -> None:
        if self.verbose:
            cli.console.print(
                "\nAverage pairwise distances: ",
                end="\n\t",
            )  # pragma: nocover

    @classmethod
    def create_input_embeddings(cls, shape: tuple[int, ...]) -> torch.nn.Embedding:
        feature_shape = math.prod(shape[1:])
        embeddings = torch.nn.Embedding(shape[0], feature_shape)
        torch.nn.init.normal_(embeddings.weight)
        return embeddings

    def forward(self, _: Any) -> torch.Tensor:
        outputs = []
        for reconstruction in self.reconstructions:
            output = reconstruction(self.inputs_embedding.weight.reshape(self.shape))
            reconstruction.zero_grad()
            outputs.append(output)
        return torch.stack(outputs)

    def calculate_loss(self, outputs: torch.Tensor) -> torch.Tensor:
        outputs = torch.transpose(outputs, 1, 0).contiguous()
        pairwise_distances = torch.cdist(outputs, outputs)
        distance_total = pairwise_distances.mean()
        loss = -distance_total
        if self.verbose:
            cli.console.print(
                f"{distance_total.item():.3f}",
                end=" ",
            )  # pragma: nocover
        return loss

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        outputs = self(batch)
        return self.calculate_loss(outputs)

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.Adam(
            self.inputs_embedding.parameters(),
            lr=self.learning_rate,
        )

    def extract_optimized_inputs(self) -> torch.Tensor:
        return self.inputs_embedding.weight.detach()


class EmptyDataset(Dataset[torch.Tensor]):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor([])


@dataclass
class Reconstructor(base.Reconstructor):
    n_networks: int = field(default_factory=lambda: context.config.n_networks)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.reconstructions = [
            self.pipeline.network_factory.create_network(seed=seed)
            for seed in range(self.n_networks)
        ]

    def create_queries(self, num_samples: int) -> torch.Tensor:  # noqa: ARG002
        return self.create_difficult_samples()

    def create_difficult_samples(self) -> torch.Tensor:
        shape = (self.num_samples, *self.input_shape)
        network = InputNetwork(shape, self.reconstructions)
        self.fit_inputs_network(network)
        return network.extract_optimized_inputs()

    @classmethod
    def fit_inputs_network(cls, network: InputNetwork) -> None:
        epochs = context.config.difficult_inputs_training.epochs
        trainer = Trainer(max_epochs=epochs, log_every_n_steps=1)
        dataset = EmptyDataset()
        dataloader = DataLoader(dataset)
        trainer.fit(network, dataloader)

    def run_round(self, data: DataModule) -> None:
        trainer = self.create_trainer(max_epochs=1)
        networks = [Network(reconstruction) for reconstruction in self.reconstructions]
        for network in networks:
            trainer.fit(network, data)
