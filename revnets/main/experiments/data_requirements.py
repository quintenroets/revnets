from ... import evaluations
from ...networks.base import Network
from ...reconstructions.outputs_supervision import random_inputs
from . import experiment


class Runner(experiment.Experiment):
    @classmethod
    def run_network(cls, network: Network):
        original = network.get_trained_network()

        results = {}
        # for i in range(500, 1000, 500):
        for i in range(1, 2):
            reconstructor = random_inputs.Reconstructor(original, network)
            reconstructor.dataset_kwargs = dict(num_samples=i)
            reconstruction = reconstructor.reconstruct()
            evaluation = evaluations.evaluate(original, reconstruction, network)
            results[reconstructor.name] = evaluation

        table = cls.make_table(results)
        table.show()
