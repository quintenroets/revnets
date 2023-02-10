from dataclasses import dataclass

from .. import evaluations, networks, reconstructions
from ..networks.base import Network
from ..utils import config, rank_zero_only


@dataclass
class Experiment:
    @classmethod
    def run(cls):
        cls.show()
        network: Network = networks.__dict__[config.network_name.value].Network()
        original = network.get_trained_network()

        for technique in reconstructions.get_algorithms():
            reconstruction = network.get_architecture()
            technique.reconstruct(original, reconstruction)
            evaluations.evaluate(original, reconstruction)

    @classmethod
    @rank_zero_only
    def show(cls):
        config.show()
