from dataclasses import dataclass

import torch

from ..data.mnist1d import Dataset
from ..networks.base import Network


@dataclass
class Evaluator:
    original: torch.nn.Module
    reconstruction: torch.nn.Module
    network: Network | None

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
