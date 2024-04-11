from collections.abc import Iterable
from dataclasses import dataclass

from torch import nn

from .. import mininet


@dataclass(frozen=True)
class NetworkFactory(mininet.NetworkFactory):
    input_size: int = 784
    hidden_size: int = 40

    def create_layers(self) -> Iterable[nn.Module]:
        yield nn.Flatten()
        yield from super().create_layers()
