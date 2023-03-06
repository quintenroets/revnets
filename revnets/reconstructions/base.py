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
        self.reconstruct_weights()
        return self.reconstruction

    def initialize_reconstruction(self):
        return self.network.get_architecture()

    def reconstruct_weights(self):
        pass

    @classmethod
    @property
    def name(cls):  # noqa
        base_name = "revnets.reconstructions."
        name = cls.__module__.replace(base_name, "")
        for token in "_/.":
            name = name.replace(token, " ")
        return name.capitalize()
