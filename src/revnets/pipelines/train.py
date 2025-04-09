from abc import ABC
from dataclasses import dataclass, field
from functools import cached_property
from typing import cast

import torch
from pytorch_lightning import LightningModule
from torch.nn import Sequential
from torch.utils.data import DataLoader
from torchsummary import summary

from revnets import networks
from revnets.context import context
from revnets.data import DataModule
from revnets.models import Path
from revnets.networks import NetworkFactory
from revnets.training import Trainer
from revnets.training.targets import Network

from . import base


@dataclass
class Pipeline(base.Pipeline, ABC):
    max_epochs: int = field(
        default_factory=lambda: context.config.target_network_training.epochs,
    )

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

    def run_training(self, network: LightningModule, data: DataModule) -> None:
        trainer = Trainer(max_epochs=self.max_epochs)
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

    @classmethod
    def load_all_train_inputs(cls) -> torch.Tensor:
        data = cls.load_prepared_data()
        batch_size = len(data.train)  # type: ignore[arg-type]
        dataloader = DataLoader(data.train, batch_size, shuffle=False)
        inputs, _ = next(iter(dataloader))
        return cast("torch.Tensor", inputs)

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

    def log_network_summary(self) -> None:
        if not isinstance(self.network_factory, networks.images.rnn.NetworkFactory):
            network = self.create_initialized_network()
            network = network.to(dtype=torch.float32).to(context.device)
            data = self.load_prepared_data()
            summary(network, data.input_shape)
