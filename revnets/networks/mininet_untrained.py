import torch.random

from ..data import mnist1d
from ..utils import config
from . import train
from .models import mininet


class Network(train.Network):
    @classmethod
    def get_model_module(cls):
        return mininet

    @classmethod
    def get_trained_network(cls):
        torch.manual_seed(config.manual_seed)
        return cls.get_model_module().Model()

    @classmethod
    def get_architecture(cls):
        torch.manual_seed(2 * config.manual_seed)
        return cls.get_model_module().Model()

    @classmethod
    def dataset(cls):
        return mnist1d.Dataset()
