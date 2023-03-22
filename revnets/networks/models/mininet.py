from torch import nn

from ...utils import config
from . import base


class Model(base.Model):
    def __init__(self, hidden_size=20, learning_rate=None):
        super().__init__()
        self.learning_rate = learning_rate or config.lr
        self.layer1 = nn.Linear(40, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        logits = self.layer2(x)
        return logits
