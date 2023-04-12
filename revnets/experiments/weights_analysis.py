from dataclasses import dataclass

from ..evaluations import outputs, weights
from . import experiment


@dataclass
class Experiment(experiment.Experiment):
    def run_network(self):
        technique = self.get_techniques()[0]
        reconstructor = technique.Reconstructor(self.network)
        reconstruction = reconstructor.reconstruct()

        evaluators = (weights.mae, outputs.val, weights.visualizer)
        # evaluators = (weights.mae, outputs.val)
        for evaluator_module in evaluators:
            evaluator = evaluator_module.Evaluator(reconstruction, self.network)
            evaluation = evaluator.evaluate()
            if evaluation is not None:
                print(evaluation)
