from dataclasses import dataclass

import torch

from revnets.pipelines import Pipeline

from ..data import Dataset


@dataclass
class Evaluator:
    reconstruction: torch.nn.Module
    pipeline: Pipeline | None

    def __post_init__(self) -> None:
        self.reconstruction = self.reconstruction.to(self.device)

    @property
    def original(self) -> torch.nn.Module:
        assert self.pipeline is not None
        return self.pipeline.create_trained_network().to(self.device)

    @property
    def device(self):
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def get_evaluation(self) -> str:
        evaluation = self.evaluate()
        return self.format_evaluation(evaluation)

    @classmethod
    def format_evaluation(cls, value, precision: int = 3):
        if value is None:
            result = "/"
        elif isinstance(value, float):
            result = f"{value:.{precision}e}"
        else:
            result = value
        return result

    def evaluate(self):
        raise NotImplementedError

    def get_dataset(self) -> Dataset:
        return self.network.create_dataset()
