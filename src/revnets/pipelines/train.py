from abc import ABC
from dataclasses import dataclass
from typing import cast

import torch
from pytorch_lightning import LightningModule
from torch.nn import Sequential

from revnets import training
from revnets.networks import NetworkFactory
from revnets.training import Trainer

from ..context import context
from ..data.base import Dataset
from ..models import Path
from . import base


@dataclass
class Pipeline(base.Pipeline, ABC):
    network_factory: NetworkFactory

    def create_initialized_network(self) -> Sequential:
        return self.network_factory.create_network(seed=context.config.experiment.seed)

    def create_trained_network(self) -> Sequential:
        if not self.weights_path.exists():
            self.create_trained_weights()
        return self.load_trained_network()

    def load_trained_network(self) -> Sequential:
        network = self.network_factory.create_network()
        self.load_weights(network)
        return network

    def create_trained_weights(self) -> None:
        seed = context.config.experiment.target_network_seed
        network = self.network_factory.create_network(seed)
        self.train(network)
        self.save_weights(network)

    def train(self, network: torch.nn.Module) -> None:
        trainable_network = training.Network(
            network, learning_rate=context.config.target_network_training.learning_rate
        )
        data = self.create_dataset()
        self.run_training(trainable_network, data)

    @classmethod
    def run_training(cls, network: LightningModule, data: Dataset) -> None:
        data.calibrate(network)
        trainer = Trainer(max_epochs=context.config.target_network_training.epochs)
        trainer.fit(network, data)
        trainer.test(network, data)

    @classmethod
    def create_dataset(cls) -> Dataset:
        raise NotImplementedError

    def calculate_output_size(self) -> int:
        dataset = self.create_dataset()
        dataset.prepare()
        sample = dataset.train_val_dataset[0][0]
        inputs = sample.unsqueeze(0)
        model = self.create_initialized_network()
        outputs = model(inputs)[0]
        size = outputs.shape[-1]
        return cast(int, size)

    def load_weights(self, model: torch.nn.Module) -> None:
        state = torch.load(self.weights_path_str)
        model.load_state_dict(state)

    def save_weights(self, model: torch.nn.Module) -> None:
        state_dict = model.state_dict()
        torch.save(state_dict, self.weights_path_str)

    @property
    def weights_path(self) -> Path:
        config = context.config.experiment
        path: Path = (
            Path.weights
            / "trained_targets"
            / "_".join(config.pipeline)
            / str(config.target_network_seed)
        )
        path.create_parent()
        return path

    @property
    def weights_path_str(self) -> str:
        return str(self.weights_path)
