from torch import nn

from . import base


class Model(base.Model):
    def __init__(self, hidden_size=20, **kwargs):
        super().__init__(**kwargs)
        self.layer1 = nn.Linear(40, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.1)
        logits = self.layer2(x)
        return logits
