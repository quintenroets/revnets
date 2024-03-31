import torch

from . import mae


class Evaluator(mae.Evaluator):
    def calculate_distance(self):
        max_distance = max(
            self.calculate_weights_distance(original, reconstruction)
            for original, reconstruction in self.iterate_compared_layers()
        )
        return max_distance

    @classmethod
    def calculate_weights_distance(cls, original_weights, reconstructed_weights):
        distances = torch.nn.functional.l1_loss(
            original_weights, reconstructed_weights, reduction="none"
        )
        distance = distances.max()
        return distance.item()
