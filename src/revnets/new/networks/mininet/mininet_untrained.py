import torch.random

from ...data import mnist1d
from ...utils import config
from .. import train
from ..models import mininet


class Network(train.Network):
    @classmethod
    def get_model_module(cls):
        return mininet

    @classmethod
    def get_trained_network(cls):
        torch.manual_seed(config.network_seed)
        return cls.initialize_model()

    @classmethod
    def get_architecture(cls, seed: int | None = None):
        if seed is None:
            seed = 2 * config.network_seed
        if seed == config.network_seed:
            raise Exception("Don't use training seed, you are cheating")
        torch.manual_seed(seed)
        return cls.initialize_model()

    @classmethod
    def dataset(cls):
        return mnist1d.Dataset()

    @classmethod
    def initialize_model(cls):
        return cls.get_model_module().Model()
