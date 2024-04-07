from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from . import base


@dataclass(frozen=True)
class NetworkFactory(base.NetworkFactory):
    input_size: int = 40
    hidden_size: int = 20

    def create_layers(self) -> Iterable[torch.nn.Module]:
        return (
            nn.Linear(self.input_size, self.hidden_size),
            self.activation_layer,
            nn.Linear(self.hidden_size, self.output_size),
        )
