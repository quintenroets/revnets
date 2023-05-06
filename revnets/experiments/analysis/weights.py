from dataclasses import dataclass

from ...evaluations import outputs, weights
from .. import experiment


@dataclass
class Experiment(experiment.Experiment):
    def run_network(self):
        technique = self.get_techniques()[0]
        reconstructor = technique.Reconstructor(self.network)
        reconstruction = reconstructor.reconstruct()
        self.run_reconstruction(reconstruction)

    def run_reconstruction(self, reconstruction):
        evaluators = (weights.mae, outputs.val, weights.visualizer)
        # evaluators = (weights.mae, outputs.val)
        # evaluators = (weights.mae, )
        # evaluators = (weights.mae, weights.mae, weights.mae, weights.mae, weights.mae)
        for evaluator_module in evaluators:
            evaluator = evaluator_module.Evaluator(reconstruction, self.network)
            evaluation = evaluator.evaluate()
            if evaluation is not None:
                print("---------")
                print(evaluation)
