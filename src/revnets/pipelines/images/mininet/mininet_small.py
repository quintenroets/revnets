from dataclasses import dataclass

from revnets import data, networks
from revnets.networks import NetworkFactory

from ... import train


@dataclass
class Pipeline(train.Pipeline):
    network_factory: NetworkFactory = networks.images.mininet.NetworkFactory(
        hidden_size=40
    )

    @classmethod
    def load_data(cls) -> data.mnist.DataModule:
        return data.mnist.DataModule()
