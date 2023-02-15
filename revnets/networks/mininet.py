import torch.random

from ..data import mnist1d
from . import train
from .models import mininet


class Network(train.Network):
    @classmethod
    def get_trained_network(cls):
        seed = 27
        torch.manual_seed(seed)
        model = mininet.Model()
        cls.load_trained_weights(model, seed)
        return model

    @classmethod
    def get_architecture(cls):
        seed = 97
        torch.manual_seed(seed)
        return mininet.Model()

    @classmethod
    def dataset(cls):
        return mnist1d.Dataset()
