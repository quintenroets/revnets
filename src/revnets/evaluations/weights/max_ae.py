import torch

from . import mae


class Evaluator(mae.Evaluator):
    def calculate_distance(self) -> float:
        return max(
            self.calculate_weights_distance(original, reconstruction)
            for original, reconstruction in self.iterate_compared_layers()
        )

    @classmethod
    def calculate_weights_distance(
        cls, original_weights: torch.Tensor, reconstructed_weights: torch.Tensor
    ) -> float:
        distances = torch.nn.functional.l1_loss(
            original_weights, reconstructed_weights, reduction="none"
        )
        distance = distances.max()
        return distance.item()
