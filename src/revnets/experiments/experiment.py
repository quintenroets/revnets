from dataclasses import dataclass
from types import ModuleType

import cli
import numpy as np
import torch

from .. import evaluations, networks, reconstructions
from ..context import context
from ..networks.models.base import Model
from ..networks.train import Network
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
        evaluation = evaluations.evaluate(self.reconstruction, self.network)
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
    def network(self) -> Network:
        names = context.config.experiment.network_to_reconstruct
        return self.extract_module(networks, names).Network()

    @property
    def reconstructor(self) -> Reconstructor:
        names = context.config.experiment.reconstruction_technique
        return self.extract_module(reconstructions, names).Reconstructor(self.network)

    @property
    def reconstruction(self) -> Model:
        return self.reconstructor.reconstruct()

    @classmethod
    def set_seed(cls) -> None:
        torch.manual_seed(context.config.experiment.seed)
        np.random.seed(context.config.experiment.seed)
