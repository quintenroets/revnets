import torch
from torch.nn.functional import l1_loss

from . import mse


class Evaluator(mse.Evaluator):
    @classmethod
    def calculate_distance(cls, values: torch.Tensor, other: torch.Tensor) -> float:
        return l1_loss(values, other, reduction="sum").item()
