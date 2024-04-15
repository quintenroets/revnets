from dataclasses import dataclass
from types import ModuleType
from typing import Any

import cli
import numpy as np
import torch
from torch import nn
from torchsummary import summary

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
        results = self.run_experiment()
        context.results_path.yaml = results

    def run_experiment(self) -> dict[str, Any]:
        pipeline: Pipeline = extract_module(pipelines, self.config.pipeline).Pipeline()
        self.log_number_of_parameters(pipeline)
        reconstruction = self.create_reconstruction(pipeline)
        evaluation = evaluations.evaluate(reconstruction, pipeline)
        evaluation.show()
        return {"metrics": evaluation.dict(), "config": context.config.dict()}

    def log_number_of_parameters(self, pipeline: Pipeline) -> None:
        network = pipeline.create_initialized_network()
        network = network.to(dtype=torch.float32).to(context.device)
        data = pipeline.load_prepared_data()
        summary(network, data.input_shape)

    def create_reconstruction(self, pipeline: Pipeline) -> nn.Module:
        module = extract_module(reconstructions, self.config.reconstruction_technique)
        reconstructor: Reconstructor = module.Reconstructor(pipeline)
        return reconstructor.create_reconstruction()


def set_seed() -> None:
    torch.manual_seed(context.config.experiment.seed)
    np.random.seed(context.config.experiment.seed)


def extract_module(module: ModuleType, names: list[str]) -> ModuleType:
    for name in names:
        module = getattr(module, name)
    return module
