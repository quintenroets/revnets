import pytorch_lightning as pl
import torch.optim as optim
import torch.random
from torch import nn

from ..data import mnist1d
from . import train


class Network(train.Network):
    @classmethod
    def get_trained_network(cls):
        torch.manual_seed(27)
        model = Model()
        Network.train(model)
        return model

    @classmethod
    def get_architecture(cls):
        torch.manual_seed(97)
        return Model()

    @classmethod
    def dataset(cls):
        return mnist1d.Dataset()


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(40, 20)
        self.layer2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        logits = self.layer2(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.001)
