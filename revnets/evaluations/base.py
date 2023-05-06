from dataclasses import dataclass

import torch

from ..data import Dataset
from ..networks.base import Network


@dataclass
class Evaluator:
    reconstruction: torch.nn.Module
    network: Network | None

    def __post_init__(self):
        if self.network is not None:
            self.original: torch.nn.Module = self.network.trained_network
            self.original = self.original.to(self.device)
        self.reconstruction = self.reconstruction.to(self.device)

    @property
    def device(self):
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def get_evaluation(self) -> str:
        evaluation = self.evaluate()
        return self.format_evaluation(evaluation)

    @classmethod
    def format_evaluation(cls, value, precision=3):
        if value is None:
            result = "/"
        elif isinstance(value, float):
            result = f"{value:.{precision}f}"
        else:
            result = value
        return result

    def evaluate(self):
        raise NotImplementedError

    def get_dataset(self) -> Dataset:
        return self.network.dataset()  # noqa
