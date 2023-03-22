import torch
import torch.optim as optim

from ...utils import NamedClass


class Model(torch.nn.Module, NamedClass):
    @classmethod
    def get_base_name(cls):
        return Model.__module__

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
