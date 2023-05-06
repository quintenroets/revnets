from torch import nn

from . import base


class Model(base.Model):
    def __init__(
        self, input_size=40, hidden_size1=20, hidden_size2=10, output_size=10, **kwargs
    ):
        super().__init__(**kwargs)
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        x = nn.functional.relu(x)
        x = self.layer3(x)
        return x
