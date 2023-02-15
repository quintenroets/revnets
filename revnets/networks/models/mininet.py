import torch.nn
import torch.optim as optim
from torch import nn

from ...utils import config


class Model(torch.nn.Module):
    def __init__(self, hidden_size=20):
        super().__init__()
        self.layer1 = nn.Linear(40, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        logits = self.layer2(x)
        return logits

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=config.lr)

    @classmethod
    @property
    def name(cls):
        return cls.__module__.replace(".", "_")
