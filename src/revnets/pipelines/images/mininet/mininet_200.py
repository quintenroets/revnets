from revnets import networks
from revnets.networks import NetworkFactory

from . import mininet_small


class Pipeline(mininet_small.Pipeline):
    @classmethod
    def create_network_factory(cls) -> NetworkFactory:
        return networks.images.mininet.NetworkFactory(hidden_size=200)
