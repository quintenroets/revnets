from collections.abc import Iterable
from dataclasses import dataclass

from torch import nn

from revnets.networks.images import mediumnet


@dataclass
class NetworkFactory(mediumnet.NetworkFactory):
    input_shape: tuple[int, ...] = 1, 28, 28

    def create_layers(self) -> Iterable[nn.Module]:
        yield from (
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=8),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4, out_channels=10, kernel_size=3),
            nn.Flatten(),
        )
