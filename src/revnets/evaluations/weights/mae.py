import torch

from . import mse


class Evaluator(mse.Evaluator):
    @classmethod
    def calculate_weights_distance(
        cls, original_weights: torch.Tensor, reconstructed_weights: torch.Tensor
    ) -> float:
        distance = torch.nn.functional.l1_loss(
            original_weights, reconstructed_weights, reduction="sum"
        )
        return distance.item()
