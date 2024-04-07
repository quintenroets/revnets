from revnets import networks
from revnets.networks import NetworkFactory

from .. import mininet


class Pipeline(mininet.Pipeline):
    network_factory: NetworkFactory = networks.images.mediumnet.NetworkFactory()
