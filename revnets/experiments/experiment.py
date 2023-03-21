import cli

from .. import evaluations, networks, reconstructions
from ..networks.base import Network
from ..utils import NamedClass, Path, Table, config


class Experiment(NamedClass):
    @classmethod
    def run(cls):
        config.show()
        for network_module in networks.get_networks():
            network = network_module.Network()
            cli.console.rule(network.name)
            cls.run_network(network)

    @classmethod
    def run_network(cls, network: Network):
        results = cls.get_results(network)
        table = cls.make_table(results)
        table.show()
        cls.save(results)

    @classmethod
    def get_results(cls, network: Network):
        original = network.get_trained_network()

        results = {}
        for technique in reconstructions.get_algorithms():
            reconstructor = technique.Reconstructor(original, network)
            reconstruction = reconstructor.reconstruct()
            evaluation = evaluations.evaluate(original, reconstruction, network)
            results[reconstructor.name] = evaluation
        return results

    @classmethod
    def make_table(cls, results):
        table = Table(show_lines=True)
        table.add_column("Technique", style="cyan", max_width=20, overflow="fold")

        metrics = next(iter(results.values())).metric_names()
        for name in metrics:
            table.add_column(name, style="magenta", max_width=13)
        for name, metrics in results.items():
            name = str(name)
            values = metrics.values()
            table.add_row(name, *values)

        return table

    @classmethod
    def get_base_name(cls):
        return Experiment.__module__

    @classmethod
    def save(cls, results):
        cls.results_path.yaml = cls.serialize_results(results)

    @classmethod
    @property
    def results_path(cls):  # noqa
        path = Path.results / cls.name / "results.yaml"
        path = path.with_nonexistent_name()
        return path

    @classmethod
    def serialize_results(cls, results: dict[str, evaluations.Evaluation]):
        return {str(k): v.dict() for k, v in results.items()}
