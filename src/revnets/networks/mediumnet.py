from collections.abc import Iterable
from dataclasses import dataclass

from torch import nn

from . import base


@dataclass(frozen=True)
class NetworkFactory(base.NetworkFactory):
    input_size: int = 40
    hidden_size1: int = 20
    hidden_size2: int = 10
    output_size: int = 10

    def create_layers(self) -> Iterable[nn.Module]:
        return (
            nn.Linear(self.input_size, self.hidden_size1),
            self.activation_layer,
            nn.Linear(self.hidden_size1, self.hidden_size2),
            self.activation_layer,
            nn.Linear(self.hidden_size2, self.output_size),
        )
