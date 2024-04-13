from dataclasses import dataclass
from functools import cached_property
from typing import Any, cast

import torch

from revnets.context import context
from revnets.pipelines import Pipeline


@dataclass
class Evaluator:
    reconstruction: torch.nn.Module
    pipeline: Pipeline

    def __post_init__(self) -> None:
        self.reconstruction = self.reconstruction.to(context.device)

    @cached_property
    def original(self) -> torch.nn.Module:
        return self.pipeline.create_target_network().to(context.device)

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
