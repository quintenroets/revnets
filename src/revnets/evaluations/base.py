from dataclasses import dataclass
from functools import cached_property
from typing import Any, cast

import torch

from revnets.context import context
from revnets.pipelines import Pipeline

from ..data import DataModule


@dataclass
class Evaluator:
    reconstruction: torch.nn.Module
    pipeline: Pipeline | None

    def __post_init__(self) -> None:
        self.reconstruction = self.reconstruction.to(context.device)

    @cached_property
    def original(self) -> torch.nn.Module:
        assert self.pipeline is not None
        return self.pipeline.create_trained_network().to(context.device)

    def get_evaluation(self) -> str:
        evaluation = self.evaluate()
        return self.format_evaluation(evaluation)

    @classmethod
    def format_evaluation(
        cls, value: float | tuple[float, ...] | None, precision: int = 3
    ) -> str:
        if value is None:
            result = "/"
        elif isinstance(value, float):
            result = f"{value:.{precision}e}"
        else:
            result = cast(str, value)
        return result

    def evaluate(self) -> Any:
        raise NotImplementedError

    def load_data(self) -> DataModule:
        assert self.pipeline is not None
        return self.pipeline.load_prepared_data()
