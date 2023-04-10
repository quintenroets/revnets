import torch

from ...utils import config
from . import mininet


class Network(mininet.Network):
    @classmethod
    def get_architecture(cls):
        torch.manual_seed(2 * config.manual_seed)
        return cls.get_model_module().Model(hidden_size=40)
