from dataclasses import dataclass, fields
from enum import Enum

import torch
from simple_classproperty import classproperty


class Phase(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    SILENT = "silent"


class MetricName(Enum):
    ACCURACY = "accuracy"
    LOSS = "loss"


@dataclass
class LogMetrics:
    def dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.names}

    @classmethod
    @classproperty
    def names(cls) -> list[str]:
        return [field.name for field in fields(cls)]


@dataclass
class Metrics(LogMetrics):
    accuracy: float
    loss: torch.Tensor
