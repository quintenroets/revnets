import torch

from ...utils import config
from . import mininet


class Network(mininet.Network):
    @classmethod
    def get_architecture(cls, seed=None):
        if seed is None:
            seed = 2 * config.network_seed
        torch.manual_seed(seed)
        return cls.get_model_module().Model(hidden_size=40)
