from dataclasses import dataclass

import cli

from .. import evaluations, networks, reconstructions
from ..networks.base import Network
from ..utils import Path, config, rank_zero_only
from ..utils.table import Table


def get_technique_name(technique):
    techniques_path = Path(reconstructions.__file__).parent
    technique_path = Path(technique.__file__)
    sub_name = technique_path.relative_to(techniques_path).with_suffix("")
    return str(sub_name).capitalize()


@dataclass
class Experiment:
    @classmethod
    def run(cls):
        # cls.show()
        for name in config.network_names:
            network: Network = networks.__dict__[name.value].Network()
            cli.console.rule(name.value)
            cls.run_network(network)

    @classmethod
    def run_network(cls, network: Network):
        original = network.get_trained_network()

        results = {}
        for technique in reconstructions.get_algorithms():
            reconstruction = network.get_architecture()
            technique.reconstruct(original, reconstruction, network)
            evaluation_metrics = evaluations.evaluate(original, reconstruction, network)
            name = get_technique_name(technique)
            results[name] = evaluation_metrics

        table = cls.make_table(results)
        table.show()

    @classmethod
    def make_table(cls, results):
        table = Table()
        table.add_column("Technique", style="cyan", no_wrap=True)
        metrics = evaluations.Evaluation.metric_names()
        for name in metrics:
            table.add_column(name, style="magenta", no_wrap=True)
        for name, metrics in results.items():
            values = metrics.get_value_list()
            table.add_row(name, *values)

        return table

    @classmethod
    @rank_zero_only
    def show(cls):
        config.show()
