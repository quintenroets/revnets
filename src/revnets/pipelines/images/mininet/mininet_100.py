from revnets import networks
from revnets.networks import NetworkFactory

from . import mininet_small


class Pipeline(mininet_small.Pipeline):
    network_factory: NetworkFactory = networks.images.mininet.NetworkFactory(
        hidden_size=100
    )
