from dataclasses import dataclass

from torch.nn import Module

from ...evaluations import outputs, weights
from .. import experiment


@dataclass
class Experiment(experiment.Experiment):
    def run(self) -> None:
        reconstruction = self.reconstructor.create_reconstruction()
        self.run_reconstruction(reconstruction)

    def run_reconstruction(self, reconstruction: Module) -> None:
        evaluators = (weights.mae, outputs.val, weights.visualizer)
        for evaluator_module in evaluators:
            evaluator = evaluator_module.Evaluator(reconstruction, self.pipeline)
            evaluation = evaluator.evaluate()
            if evaluation is not None:
                print("---------")
                print(evaluation)
