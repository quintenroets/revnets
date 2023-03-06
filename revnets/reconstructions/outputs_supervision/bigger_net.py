import torch

from ...networks.models import mininet
from . import base


class Reconstructor(base.Reconstructor):
    @classmethod
    def initialize_reconstruction(cls):
        seed = 97
        torch.manual_seed(seed)
        return mininet.Model(hidden_size=256)
