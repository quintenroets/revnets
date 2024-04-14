from dataclasses import dataclass

from revnets import networks
from revnets.networks import NetworkFactory

from . import mininet


@dataclass
class Pipeline(mininet.Pipeline):
    @classmethod
    def create_network_factory(cls) -> NetworkFactory:
        return networks.mininet.NetworkFactory(hidden_size=40)
