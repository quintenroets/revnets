from collections.abc import Iterable
from dataclasses import dataclass

from torch import nn

from .. import mediumnet


@dataclass
class NetworkFactory(mediumnet.NetworkFactory):
    input_shape: tuple[int, ...] = 1, 28, 28

    def create_layers(self) -> Iterable[nn.Module]:
        yield from (
            # 1 x 28 x 28
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=8),
            nn.LeakyReLU(),
            # 6 x 6 x 6
            # nn.Conv2d(in_channels=6, out_channels=6, kernel_size=5, stride=2),
            # nn.Tanh(),
            # 6 x 4 x 4
            # 96
            nn.Conv2d(in_channels=4, out_channels=10, kernel_size=3),
            nn.Flatten(),
        )
