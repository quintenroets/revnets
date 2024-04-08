from dataclasses import dataclass

from ... import reconstructions
from .. import experiment


@dataclass
class Experiment(experiment.Experiment):
    def run(self) -> None:
        technique = reconstructions.outputs_supervision.iterative_sampling.analysis
        reconstructor = technique.Reconstructor(self.pipeline)
        reconstructor.create_reconstruction()
