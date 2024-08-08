from dataclasses import dataclass

from revnets.data import mnist1d
from revnets.networks import NetworkFactory, mininet
from revnets.pipelines import train


@dataclass
class Pipeline(train.Pipeline):
    @classmethod
    def create_network_factory(cls) -> NetworkFactory:
        return mininet.NetworkFactory()

    @classmethod
    def load_data(cls) -> mnist1d.DataModule:
        return mnist1d.DataModule()
