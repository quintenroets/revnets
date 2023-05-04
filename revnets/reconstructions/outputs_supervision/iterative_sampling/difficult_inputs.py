import math
from dataclasses import dataclass, field
from typing import Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from revnets.utils import Trainer, config

from ..base import ReconstructModel
from . import base


class InputModel(pl.LightningModule):
    def __init__(self, shape, reconstructions, learning_rate=0.01, verbose=True):
        super().__init__()
        self.learning_rate = learning_rate
        self.inputs_embedding = self.get_input_embeddings(shape)
        self.reconstructions = torch.nn.ModuleList(reconstructions)
        self.verbose = verbose

    def on_train_start(self) -> None:
        if self.verbose:
            print("\nAverage pairwise distances: ", end="\n\t")

    @classmethod
    def get_input_embeddings(cls, shape) -> torch.nn.Embedding:
        embeddings = torch.nn.Embedding(*shape)
        torch.nn.init.normal_(embeddings.weight)
        return embeddings

    def forward(self, _) -> Any:
        outputs = []
        for reconstruction in self.reconstructions:
            output = reconstruction(self.inputs_embedding.weight)
            reconstruction.zero_grad()
            outputs.append(output)
        return torch.stack(outputs)

    def calculate_loss(self, outputs):
        outputs = torch.transpose(outputs, 1, 0).contiguous()
        pairwise_distances = torch.cdist(outputs, outputs)
        distance_total = pairwise_distances.mean()
        loss = -distance_total
        if self.verbose:
            print(f"{distance_total.item():.3f}", end=" ")
        return loss

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        return self.calculate_loss(outputs)

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(
            self.inputs_embedding.parameters(), lr=self.learning_rate
        )

    def get_optimized_inputs(self):
        return self.inputs_embedding.weight.detach()


class EmptyDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.tensor([])


@dataclass
class Reconstructor(base.Reconstructor):
    n_networks: int = config.n_networks
    models: list[ReconstructModel] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        self.reconstructions = [self.reconstruction]
        for i in range(self.n_networks - 1):
            reconstruction = self.network.get_architecture(seed=i)
            self.reconstructions.append(reconstruction)

    @property
    def feature_size(self):
        return math.prod(self.data.input_shape)

    def get_difficult_inputs(self):
        shape = (self.num_samples, self.feature_size)
        model = InputModel(shape, self.reconstructions)
        self.fit_inputs_model(model)
        return model.get_optimized_inputs()

    @classmethod
    def fit_inputs_model(cls, model, epochs=100):
        trainer = Trainer(max_epochs=epochs, log_every_n_steps=1)
        dummy_dataset = EmptyDataset()
        dummy_dataloader = DataLoader(dummy_dataset)
        trainer.fit(model, dummy_dataloader)

    def start_training(self):
        self.models = [
            ReconstructModel(reconstruction, self.network)
            for reconstruction in self.reconstructions
        ]
        super().start_training()

    def run_round(self):
        for model in self.models:
            self.model = model
            self.train_model(self.data)
            self.check_randomize()

        if not self.last_round:
            self.add_difficult_samples()
