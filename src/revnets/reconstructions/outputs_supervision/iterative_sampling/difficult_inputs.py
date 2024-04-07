import math
from dataclasses import dataclass, field
from typing import Any

import torch
from pytorch_lightning import LightningModule
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from revnets.context import context
from revnets.training import Trainer

from ..base import ReconstructNetwork
from . import base


class InputNetwork(LightningModule):
    def __init__(
        self,
        shape: tuple[int, int],
        reconstructions: list[torch.nn.Sequential],
        learning_rate: float = 0.01,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.inputs_embedding = self.get_input_embeddings(shape)
        self.reconstructions = torch.nn.ModuleList(reconstructions)
        self.verbose = verbose

    def on_train_start(self) -> None:
        if self.verbose:
            print("\nAverage pairwise distances: ", end="\n\t")

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
            print(f"{distance_total.item():.3f}", end=" ")
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
    networks: list[ReconstructNetwork] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.reconstructions = [self.reconstruction]
        for i in range(self.n_networks - 1):
            reconstruction = self.network.get_architecture(seed=i)
            self.reconstructions.append(reconstruction)

    @property
    def feature_size(self) -> int:
        return math.prod(self.data.input_shape)

    def get_difficult_inputs(self) -> torch.Tensor:
        shape = (self.num_samples, self.feature_size)
        network = InputNetwork(shape, self.reconstructions)
        self.fit_inputs_network(network)
        return network.get_optimized_inputs()

    @classmethod
    def fit_inputs_network(cls, network: InputNetwork, epochs: int = 100) -> None:
        trainer = Trainer(max_epochs=epochs, log_every_n_steps=1)
        dummy_dataset = EmptyDataset()
        dummy_dataloader = DataLoader(dummy_dataset)
        trainer.fit(network, dummy_dataloader)

    def start_training(self) -> None:
        self.networks = [
            ReconstructNetwork(reconstruction, self.network)
            for reconstruction in self.reconstructions
        ]
        super().start_training()

    def run_round(self) -> None:
        for network in self.networks:
            self.network = network
            self.train_model(self.data)

        if not self.is_last_round:
            self.add_difficult_samples()
