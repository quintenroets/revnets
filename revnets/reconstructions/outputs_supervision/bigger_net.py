import torch

from ...networks.models import mininet
from ...utils import config
from . import base


class Reconstructor(base.Reconstructor):
    @classmethod
    def initialize_reconstruction(cls):
        torch.manual_seed(config.manual_seed * 2)
        return mininet.Model(hidden_size=256)
