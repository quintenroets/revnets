import torch
from torch.nn.functional import l1_loss

from . import mae


class Evaluator(mae.Evaluator):
    def calculate_total_distance(self) -> float:
        return max(
            self.calculate_distance(original, reconstruction)
            for original, reconstruction in self.iterate_compared_layers()
        )

    @classmethod
    def calculate_distance(cls, values: torch.Tensor, other: torch.Tensor) -> float:
        return l1_loss(values, other, reduction="none").max().item()
