from dataclasses import dataclass

from revnets import networks
from revnets.networks import NetworkFactory

from .mininet import mininet_small


@dataclass
class Pipeline(mininet_small.Pipeline):
    max_epochs: int = 10

    @classmethod
    def create_network_factory(cls) -> NetworkFactory:
        return networks.images.cnn.mini.NetworkFactory()
