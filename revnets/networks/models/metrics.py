from dataclasses import dataclass, fields
from enum import Enum

import torch


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
    def dict(self):
        return {name: getattr(self, name) for name in self.names}

    @classmethod
    @property
    def names(cls):
        return [field.name for field in fields(cls)]


@dataclass
class Metrics(LogMetrics):
    accuracy: float
    loss: torch.Tensor
