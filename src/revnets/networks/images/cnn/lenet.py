from collections.abc import Iterable
from dataclasses import dataclass

from torch import nn

from .. import mediumnet


@dataclass
class NetworkFactory(mediumnet.NetworkFactory):
    hidden_size1: int = 120
    hidden_size2: int = 84

    def create_layers(self) -> Iterable[nn.Module]:
        yield from (
            # 28 x 28 x 1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            # 24 x 24 x 6
            nn.AvgPool2d(kernel_size=2, stride=2),
            # 12 x 12 x 6
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            # 8 x 8 x 16
            nn.AvgPool2d(kernel_size=2, stride=2),
            # 4 x 4 x 16
            nn.Flatten(),
            # 256
            nn.Linear(in_features=256, out_features=self.hidden_size1),
            nn.Tanh(),
            nn.Linear(in_features=self.hidden_size1, out_features=self.hidden_size2),
            nn.Tanh(),
            nn.Linear(in_features=self.hidden_size2, out_features=self.output_size),
        )
