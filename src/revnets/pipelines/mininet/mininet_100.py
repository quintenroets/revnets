from dataclasses import dataclass

from revnets import networks
from revnets.networks import NetworkFactory

from . import mininet


@dataclass
class Pipeline(mininet.Pipeline):
    network_factory: NetworkFactory = networks.mininet.NetworkFactory(hidden_size=100)
