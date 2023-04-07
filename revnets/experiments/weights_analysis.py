from dataclasses import dataclass

from .. import networks
from ..evaluations import outputs, weights
from ..reconstructions.outputs_supervision import random_inputs
from . import experiment


@dataclass
class Experiment(experiment.Experiment):
    @classmethod
    def get_networks(cls):
        return (networks.mininet.mininet,)

    def run_network(self):
        reconstructor = random_inputs.Reconstructor(self.network)
        reconstruction = reconstructor.reconstruct()

        mae_evaluator = weights.mae.Evaluator(reconstruction, self.network)
        mae = mae_evaluator.evaluate()
        print(mae)
        return

        outputs_mae = outputs.val.Evaluator(
            reconstruction, self.network
        ).get_evaluation()
        print(outputs_mae)

        evaluator = weights.visualizer.Evaluator(reconstruction, self.network)
        evaluator.evaluate()
        # input("Enter to exit: ")
