from collections.abc import Iterable
from dataclasses import dataclass

from torch import nn

from . import mediumnet


@dataclass
class NetworkFactory(mediumnet.NetworkFactory):
    input_size: int = 784
    hidden_size1: int = 512
    hidden_size2: int = 256

    def create_layers(self) -> Iterable[nn.Module]:
        yield nn.Flatten()
        yield from super().create_layers()
