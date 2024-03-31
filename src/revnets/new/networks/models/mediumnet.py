from torch import nn

from . import base


class Model(base.Model):
    def __init__(
        self,
        input_size: int = 40,
        hidden_size1: int = 20,
        hidden_size2: int = 10,
        output_size: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation_function(x)
        x = self.layer2(x)
        x = self.activation_function(x)
        x = self.layer3(x)
        return x
