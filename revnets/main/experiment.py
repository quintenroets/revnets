from dataclasses import dataclass

import cli

from .. import evaluations, networks, reconstructions
from ..networks.base import Network
from ..utils import config
from ..utils.table import Table


@dataclass
class Experiment:
    @classmethod
    def run(cls):
        config.show()
        for name in config.network_names:
            network: Network = networks.__dict__[name.value].Network()
            cli.console.rule(name.value)
            cls.run_network(network)

    @classmethod
    def run_network(cls, network: Network):
        original = network.get_trained_network()

        results = {}
        for technique in reconstructions.get_algorithms():
            reconstructor = technique.Reconstructor(original, network)
            reconstruction = reconstructor.reconstruct()
            evaluation = evaluations.evaluate(original, reconstruction, network)
            results[reconstructor.name] = evaluation

        table = cls.make_table(results)
        table.show()

    @classmethod
    def make_table(cls, results):
        table = Table(show_lines=True)
        table.add_column("Technique", style="cyan", max_width=20, overflow="fold")

        metrics = next(iter(results.values())).metric_names()
        for name in metrics:
            table.add_column(name, style="magenta", max_width=13)
        for name, metrics in results.items():
            values = metrics.values()
            table.add_row(name, *values)

        return table
