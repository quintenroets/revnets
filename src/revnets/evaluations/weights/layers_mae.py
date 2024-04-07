from collections.abc import Iterator

import torch

from . import mae
from .standardize import extract_layer_weights, generate_layers


class Evaluator(mae.Evaluator):
    def iterate_compared_layers(
        self, device: torch.device | None = None
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        original_layers = generate_layers(self.original)
        reconstruction_layers = generate_layers(self.reconstruction)
        for original, reconstruction in zip(original_layers, reconstruction_layers):
            original_weights = extract_layer_weights(original, device)
            reconstruction_weights = extract_layer_weights(reconstruction, device)
            yield original_weights, reconstruction_weights

    def calculate_distance(self) -> tuple[float, ...]:
        return tuple(
            self.calculate_weights_distance(original, reconstructed)
            for original, reconstructed in self.iterate_compared_layers()
        )

    @classmethod
    def calculate_weights_distance(
        cls, original_weights: torch.Tensor, reconstructed_weights: torch.Tensor
    ) -> float:
        distance = torch.nn.functional.l1_loss(original_weights, reconstructed_weights)
        return distance.item()

    @classmethod
    def format_evaluation(cls, value: tuple[float, ...], precision: int = 3) -> str:  # type: ignore[override]
        if value:
            values = (
                super(Evaluator, cls).format_evaluation(layer_value)
                for layer_value in value
            )
            formatted_value = ", ".join(values)
        else:
            formatted_value = "/"
        return formatted_value
