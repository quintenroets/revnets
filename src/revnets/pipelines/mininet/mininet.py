from dataclasses import dataclass

from revnets.networks import NetworkFactory, mininet

from ...data import mnist1d
from .. import train


@dataclass
class Pipeline(train.Pipeline):
    network_factory: NetworkFactory = mininet.NetworkFactory

    @classmethod
    def create_dataset(cls):
        return mnist1d.Dataset()
