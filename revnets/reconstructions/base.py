from dataclasses import dataclass

import torch.nn

from ..networks.mininet import Network
from ..utils import NamedClass


@dataclass
class Reconstructor(NamedClass):
    original: torch.nn.Module
    network: Network
    reconstruction: torch.nn.Module = None

    def reconstruct(self):
        self.reconstruction = self.initialize_reconstruction()
        self.reconstruct_weights()
        return self.reconstruction

    def initialize_reconstruction(self):
        return self.network.get_architecture()

    def reconstruct_weights(self):
        pass

    @classmethod
    def get_base_name(cls):
        return Reconstructor.__module__
