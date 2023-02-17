from dataclasses import dataclass

import torch.nn

from ..networks.mininet import Network


@dataclass
class Reconstructor:
    original: torch.nn.Module
    network: Network
    reconstruction: torch.nn.Module = None

    def reconstruct(self):
        self.reconstruction = self.initialize_reconstruction()
        self.train()
        return self.reconstruction

    def initialize_reconstruction(self):
        return self.network.get_architecture()

    def train(self):
        pass
