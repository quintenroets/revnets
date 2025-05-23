from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any

import cli
import torch
from torch import nn

from revnets import evaluations, pipelines, reconstructions
from revnets.context import Experiment as Config
from revnets.context import context
from revnets.pipelines import Pipeline

if TYPE_CHECKING:
    from revnets.reconstructions import Reconstructor  # pragma: nocover


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
        results = self.run_experiment()
        context.results_path.yaml = results

    def run_experiment(self) -> dict[str, Any]:
        pipeline: Pipeline = extract_module(pipelines, self.config.pipeline).Pipeline()
        pipeline.log_network_summary()
        reconstruction = self.create_reconstruction(pipeline)
        evaluation = evaluations.evaluate(reconstruction, pipeline)
        evaluation.show()
        return {"metrics": evaluation.dict(), "config": context.config.dict()}

    def create_reconstruction(self, pipeline: Pipeline) -> nn.Module:
        module = extract_module(reconstructions, self.config.reconstruction_technique)
        reconstructor: Reconstructor = module.Reconstructor(pipeline)
        return reconstructor.create_reconstruction()


def set_seed() -> None:
    torch.manual_seed(context.config.experiment.seed)


def extract_module(module: ModuleType, names: list[str]) -> ModuleType:
    for name in names:
        module = getattr(module, name)
    return module
