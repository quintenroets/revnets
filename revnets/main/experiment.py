from dataclasses import dataclass

import cli

from .. import evaluations, networks, reconstructions
from ..networks.base import Network
from ..utils import Path, config
from ..utils.table import Table


def get_technique_name(technique):
    techniques_path = Path(reconstructions.__file__).parent
    technique_path = Path(technique.__file__)
    sub_name = technique_path.relative_to(techniques_path).with_suffix("")
    return str(sub_name).capitalize().replace("_", " ")


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
            reconstruction = technique.Reconstructor(original, network).reconstruct()
            evaluation_metrics = evaluations.evaluate(original, reconstruction, network)
            name = get_technique_name(technique)
            results[name] = evaluation_metrics

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
