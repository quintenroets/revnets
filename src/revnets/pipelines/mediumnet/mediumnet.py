from revnets import networks
from revnets.networks import NetworkFactory

from ..mininet import mininet


class Pipeline(mininet.Pipeline):
    @classmethod
    def create_network_factory(cls) -> NetworkFactory:
        return networks.mediumnet.NetworkFactory()
