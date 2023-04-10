from dataclasses import dataclass

from .. import networks
from ..evaluations import outputs, weights
from ..reconstructions import outputs_supervision
from . import experiment


@dataclass
class Experiment(experiment.Experiment):
    @classmethod
    def get_networks(cls):
        return (networks.mininet.mininet,)
        return (networks.mininet.mininet_bigger_reconstruction,)

    def run_network(self):
        reconstructor = outputs_supervision.iterative_sampling.Reconstructor(
            self.network
        )
        # reconstructor = outputs_supervision.random_inputs.Reconstructor(self.network)
        reconstruction = reconstructor.reconstruct()

        evaluators = (weights.mae, outputs.val, weights.visualizer)
        # evaluators = (weights.mae, outputs.val)
        for evaluator_module in evaluators:
            evaluator = evaluator_module.Evaluator(reconstruction, self.network)
            evaluation = evaluator.evaluate()
            if evaluation is not None:
                print(evaluation)
