import torch
import torch.optim as optim

from ...utils import NamedClass, config


class Model(torch.nn.Module, NamedClass):
    def __init__(self, learning_rate=None):
        super().__init__()
        self.learning_rate = learning_rate or config.lr

    @classmethod
    def get_base_name(cls):
        return Model.__module__

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
