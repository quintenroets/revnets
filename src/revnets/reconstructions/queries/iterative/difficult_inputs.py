import math
from dataclasses import dataclass
from typing import Any

import torch
from pytorch_lightning import LightningModule
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from revnets.context import context
from revnets.training import Trainer
from revnets.training.reconstructions import Network

from ..base import DataModule
from . import base


class InputNetwork(LightningModule):
    def __init__(
        self,
        shape: tuple[int, int],
        reconstructions: list[torch.nn.Sequential],
        learning_rate: float = 0.01,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.inputs_embedding = self.get_input_embeddings(shape)
        self.reconstructions = torch.nn.ModuleList(reconstructions)
        self.verbose = verbose

    def on_train_start(self) -> None:
        if self.verbose:
            print("\nAverage pairwise distances: ", end="\n\t")  # pragma: nocover

    @classmethod
    def get_input_embeddings(cls, shape: tuple[int, int]) -> torch.nn.Embedding:
        embeddings = torch.nn.Embedding(*shape)
        torch.nn.init.normal_(embeddings.weight)
        return embeddings

    def forward(self, _: Any) -> torch.Tensor:
        outputs = []
        for reconstruction in self.reconstructions:
            output = reconstruction(self.inputs_embedding.weight)
            reconstruction.zero_grad()
            outputs.append(output)
        return torch.stack(outputs)

    def calculate_loss(self, outputs: torch.Tensor) -> torch.Tensor:
        outputs = torch.transpose(outputs, 1, 0).contiguous()
        pairwise_distances = torch.cdist(outputs, outputs)
        distance_total = pairwise_distances.mean()
        loss = -distance_total
        if self.verbose:
            print(f"{distance_total.item():.3f}", end=" ")  # pragma: nocover
        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        return self.calculate_loss(outputs)

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.Adam(
            self.inputs_embedding.parameters(), lr=self.learning_rate
        )

    def get_optimized_inputs(self) -> torch.Tensor:
        return self.inputs_embedding.weight.detach()


class EmptyDataset(Dataset[torch.Tensor]):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor([])


@dataclass
class Reconstructor(base.Reconstructor):
    n_networks: int = context.config.n_networks

    def __post_init__(self) -> None:
        super().__post_init__()
        self.reconstructions = [
            self.pipeline.network_factory.create_network(seed=seed)
            for seed in range(self.n_networks)
        ]

    @property
    def feature_size(self) -> int:
        return math.prod(self.input_shape)

    def create_difficult_samples(self) -> torch.Tensor:
        shape = (self.num_samples, self.feature_size)
        network = InputNetwork(shape, self.reconstructions)
        self.fit_inputs_network(network)
        return network.get_optimized_inputs()

    @classmethod
    def fit_inputs_network(cls, network: InputNetwork) -> None:
        max_epochs = context.config.max_difficult_inputs_epochs
        trainer = Trainer(max_epochs=max_epochs, log_every_n_steps=1)
        dataset = EmptyDataset()
        dataloader = DataLoader(dataset)
        trainer.fit(network, dataloader)

    def run_round(self, data: DataModule) -> None:
        trainer = self.create_trainer()
        networks = [Network(reconstruction) for reconstruction in self.reconstructions]
        for network in networks:
            trainer.fit(network, data)
