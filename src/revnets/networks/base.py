from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch.nn import Sequential

from ..context import context
from ..utils import NamedClass


@dataclass
class NetworkFactory(NamedClass):
    activation_layer: torch.nn.Module = context.activation_layer
    output_size: int = 10

    def create_network(self, seed: int | None = None) -> Sequential:
        if seed is not None:
            torch.manual_seed(seed)
        layers = self.create_layers()
        return Sequential(*layers)

    def create_layers(self) -> Iterable[torch.nn.Module]:
        raise NotImplementedError

    @classmethod
    def get_base_name(cls):
        return Model.__module__
