from abc import ABC
from dataclasses import dataclass
from functools import cached_property

import torch
from pytorch_lightning import LightningModule
from torch.nn import Sequential

from revnets.data import DataModule
from revnets.networks import NetworkFactory
from revnets.training import Trainer
from revnets.training.targets import Network

from ..context import context
from ..models import Path
from . import base


@dataclass
class Pipeline(base.Pipeline, ABC):
    def create_initialized_network(self) -> Sequential:
        return self.network_factory.create_network(seed=context.config.experiment.seed)

    @cached_property
    def network_factory(self) -> NetworkFactory:
        return self.create_network_factory()

    def create_network_factory(self) -> NetworkFactory:
        raise NotImplementedError  # pragma: nocover

    @cached_property
    def target(self) -> Sequential:
        return self.create_target_network()

    def create_target_network(self) -> Sequential:
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
        trainable_network = Network(network)
        data = self.load_data()
        self.run_training(trainable_network, data)

    @classmethod
    def run_training(cls, network: LightningModule, data: DataModule) -> None:
        trainer = Trainer(max_epochs=context.config.target_network_training.epochs)
        trainer.fit(network, data)
        trainer.test(network, data)

    @classmethod
    def load_data(cls) -> DataModule:
        raise NotImplementedError  # pragma: nocover

    @classmethod
    def load_prepared_data(cls) -> DataModule:
        data = cls.load_data()
        data.prepare_data()
        data.setup("train")
        return data

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
            / "_".join(self.relative_module)
            / str(config.target_network_seed)
        )
        path.create_parent()
        return path

    @property
    def weights_path_str(self) -> str:
        return str(self.weights_path)
