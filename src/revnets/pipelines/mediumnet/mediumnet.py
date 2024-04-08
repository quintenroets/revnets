from revnets import networks
from revnets.networks import NetworkFactory

from ..mininet import mininet


class Pipeline(mininet.Pipeline):
    network_factory: NetworkFactory = networks.mediumnet.NetworkFactory()
