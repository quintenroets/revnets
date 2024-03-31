from torch import nn

from . import base


class Model(base.Model):
    def __init__(self, input_size: int = 40, hidden_size: int = 20, **kwargs) -> None:
        super().__init__(**kwargs)
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation_function(x)
        logits = self.layer2(x)
        return logits
