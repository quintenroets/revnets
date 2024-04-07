from dataclasses import dataclass
from types import ModuleType

import cli
import numpy as np
import torch

from revnets import pipelines
from revnets.pipelines import Pipeline

from .. import evaluations, reconstructions
from ..context import context
from ..reconstructions import Reconstructor
from ..utils import NamedClass


@dataclass
class Experiment(NamedClass):
    """
    Reverse engineer internal parameters of black box neural networks.
    """

    def run(self) -> None:
        self.set_seed()
        cli.console.rule(context.config.experiment.title)
        reconstruction = self.reconstructor.create_reconstruction()
        evaluation = evaluations.evaluate(reconstruction, self.pipeline)
        evaluation.show()
        context.results_path.yaml = evaluation.dict()

    @classmethod
    def get_base_name(cls):
        return Experiment.__module__

    @classmethod
    def extract_module(cls, module: ModuleType, names: list[str]) -> ModuleType:
        for name in names:
            module = getattr(module, name)
        return module

    @property
    def pipeline(self) -> Pipeline:
        names = context.config.experiment.pipeline
        return self.extract_module(pipelines, names).Pipeline()

    @property
    def reconstructor(self) -> Reconstructor:
        names = context.config.experiment.reconstruction_technique
        return self.extract_module(reconstructions, names).Reconstructor(self.pipeline)

    @classmethod
    def set_seed(cls) -> None:
        torch.manual_seed(context.config.experiment.seed)
        np.random.seed(context.config.experiment.seed)
