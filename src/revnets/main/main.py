from dataclasses import dataclass
from types import ModuleType

import cli
import numpy as np
import torch

from revnets import pipelines
from revnets.models import Experiment as Config
from revnets.pipelines import Pipeline

from .. import evaluations, reconstructions
from ..context import context
from ..reconstructions import Reconstructor


@dataclass
class Experiment:
    """
    Reverse engineer internal parameters of black box neural networks.
    """

    @property
    def config(self) -> Config:
        return context.config.experiment

    def run(self) -> None:
        set_seed()
        cli.console.rule(self.config.title)
        self.run_experiment()

    def run_experiment(self) -> None:
        reconstruction = self.reconstructor.create_reconstruction()
        evaluation = evaluations.evaluate(reconstruction, self.pipeline)
        evaluation.show()
        context.results_path.yaml = evaluation.dict()

    @property
    def pipeline(self) -> Pipeline:
        pipeline: Pipeline = extract_module(pipelines, self.config.pipeline).Pipeline()
        return pipeline

    @property
    def reconstructor(self) -> Reconstructor:
        module = extract_module(reconstructions, self.config.reconstruction_technique)
        reconstructor: Reconstructor = module.Reconstructor(self.pipeline)
        return reconstructor


def set_seed() -> None:
    torch.manual_seed(context.config.experiment.seed)
    np.random.seed(context.config.experiment.seed)


def extract_module(module: ModuleType, names: list[str]) -> ModuleType:
    for name in names:
        module = getattr(module, name)
    return module
