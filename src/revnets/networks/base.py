from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import Module, Sequential

from revnets.context import Activation, context
from revnets.utils import NamedClass


@dataclass
class NetworkFactory(NamedClass):
    output_size: int = 10
    activation: Activation = field(
        default_factory=lambda: context.config.target_network_training.activation,
    )
    input_shape: tuple[int, ...] | None = None

    def create_activation_layer(self) -> Module:
        activation_layer: Module
        match self.activation:
            case Activation.leaky_relu:
                activation_layer = nn.LeakyReLU()
            case Activation.relu:
                activation_layer = nn.ReLU()
            case Activation.tanh:
                activation_layer = nn.Tanh()
        return activation_layer

    def create_network(self, seed: int | None = None) -> Sequential:
        if seed is not None:
            torch.manual_seed(seed)
        layers = [layer.to(dtype=context.dtype) for layer in self.create_layers()]
        return Sequential(*layers)

    def create_layers(self) -> Iterable[torch.nn.Module]:
        raise NotImplementedError  # pragma: nocover
