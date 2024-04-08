from revnets import data, networks
from revnets.networks import NetworkFactory

from ... import train


class Pipeline(train.Pipeline):
    network_factory: NetworkFactory = networks.images.mininet.NetworkFactory(
        hidden_size=40
    )

    @classmethod
    def create_dataset(cls) -> data.mnist.Dataset:
        return data.mnist.Dataset()
