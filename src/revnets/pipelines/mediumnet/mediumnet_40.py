from revnets import networks
from revnets.networks import NetworkFactory
from revnets.pipelines.mininet import mininet


class Pipeline(mininet.Pipeline):
    @classmethod
    def create_network_factory(cls) -> NetworkFactory:
        return networks.mediumnet.NetworkFactory(hidden_size1=40, hidden_size2=20)
