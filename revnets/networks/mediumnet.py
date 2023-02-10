import torch.random
from torch import nn


class Network:
    @classmethod
    def get_trained_network(cls):
        torch.manual_seed(27)
        return Model()

    @classmethod
    def get_architecture(cls):
        torch.manual_seed(97)
        return Model()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        x = nn.functional.relu(x)
        x = self.layer3(x)
        x = nn.functional.relu(x)
        logits = self.layer4(x)
        return logits
