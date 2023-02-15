from dataclasses import dataclass, fields
from enum import Enum

import torch


class Phase(Enum):
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"
    SILENT = "silent"


class MetricName(Enum):
    ACCURACY = "accuracy"
    LOSS = "loss"


@dataclass
class Metrics:
    accuracy: float
    loss: torch.Tensor

    def dict(self):
        return self.__dict__

    @classmethod
    @property
    def names(cls):
        return [field.name for field in fields(cls)]
