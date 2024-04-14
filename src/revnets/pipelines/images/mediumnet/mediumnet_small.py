from dataclasses import dataclass

from revnets import networks
from revnets.networks import NetworkFactory

from . import mediumnet


@dataclass
class Pipeline(mediumnet.Pipeline):
    @classmethod
    def create_network_factory(cls) -> NetworkFactory:
        return networks.images.mediumnet.NetworkFactory(
            hidden_size1=100, hidden_size2=50
        )
