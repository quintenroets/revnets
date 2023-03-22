from dataclasses import dataclass

from ..networks.base import Network
from ..networks.models.base import Model
from ..utils import NamedClass


@dataclass
class Reconstructor(NamedClass):
    network: Network
    reconstruction: Model = None

    def __post_init__(self):
        self.original: Model = self.network.trained_network

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
