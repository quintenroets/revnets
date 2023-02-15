from dataclasses import dataclass

import cli

from .. import evaluations, networks, reconstructions
from ..networks.base import Network
from ..utils import Path, config, rank_zero_only


def get_technique_name(technique):
    techniques_path = Path(reconstructions.__file__).parent
    technique_path = Path(technique.__file__)
    sub_name = technique_path.relative_to(techniques_path).with_suffix("")
    return str(sub_name).capitalize()


def show_name(technique):
    name = get_technique_name(technique)
    cli.console.print(f"[underline]{name}[/underline]")


@dataclass
class Experiment:
    @classmethod
    def run(cls):
        cls.show()
        network: Network = networks.__dict__[config.network_name.value].Network()
        original = network.get_trained_network()

        for technique in reconstructions.get_algorithms():
            show_name(technique)
            reconstruction = network.get_architecture()
            technique.reconstruct(original, reconstruction, network)
            evaluations.evaluate(original, reconstruction, network)

    @classmethod
    @rank_zero_only
    def show(cls):
        config.show()
