import json
from collections.abc import Iterator

import torch

from . import layers_mae


class Evaluator(layers_mae.Evaluator):
    def calculate_total_distance(self) -> dict[str, float]:  # type: ignore[override]
        return {
            name: self.calculate_distance(original, reconstructed)
            for name, original, reconstructed in self.iterate_named_compared_layers()
        }

    def iterate_named_compared_layers(
        self,
    ) -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
        keys = self.target.state_dict().keys()
        original_values = self.target.state_dict().values()
        reconstruction_values = self.reconstruction.state_dict().values()
        yield from zip(keys, original_values, reconstruction_values)

    @classmethod
    def format_evaluation(cls, value: dict[str, float], precision: int = 3) -> str:  # type: ignore[override]
        if value:
            values = {
                name: super(layers_mae.Evaluator, cls).format_evaluation(layer_value)
                for name, layer_value in value.items()
            }
            formatted_value = json.dumps(values, indent=4)
        else:
            formatted_value = "/"  # pragma: nocover
        return formatted_value
