import torch

from . import mae
from .standardize import standardize


class Evaluator(mae.Evaluator):
    def iterate_compared_layers(self, device=None):
        original_layers = standardize.get_layers(self.original)
        reconstruction_layers = standardize.get_layers(self.reconstruction)
        get_weights = standardize.order.get_layer_weights
        for original, reconstruction in zip(original_layers, reconstruction_layers):
            original_weights = get_weights(original, device)
            reconstruction_weights = get_weights(reconstruction, device)
            yield original_weights, reconstruction_weights

    def calculate_distance(self):
        return tuple(
            self.calculate_weights_distance(original, reconstructed)
            for original, reconstructed in self.iterate_compared_layers()
        )

    @classmethod
    def calculate_weights_distance(cls, original_weights, reconstructed_weights):
        distance = torch.nn.functional.l1_loss(original_weights, reconstructed_weights)
        return distance.item()

    @classmethod
    def format_evaluation(cls, value, precision=3) -> str:
        if value:
            values = (
                super(Evaluator, cls).format_evaluation(layer_value)
                for layer_value in value
            )
            formatted_value = ", ".join(values)
        else:
            formatted_value = "/"
        return formatted_value
