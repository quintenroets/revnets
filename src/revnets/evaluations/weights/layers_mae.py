from collections.abc import Iterator

import torch
from torch.nn import Module
from torch.nn.functional import l1_loss

from revnets.standardization import extract_layers
from revnets.standardization.weights import feedforward, rnn

from . import mae


def extract_weights(model: Module) -> Iterator[torch.Tensor]:
    for layer in extract_layers(model):
        if isinstance(layer.weights, feedforward.Weights):
            yield layer.weights.weights
        elif isinstance(layer.weights, rnn.Weights):
            yield layer.weights.input_to_hidden.weights
            yield layer.weights.hidden_to_hidden.weights
        else:
            raise ValueError("Unexpected weights type")  # pragma: nocover


class Evaluator(mae.Evaluator):
    def iterate_compared_layers(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        networks = self.target, self.reconstruction
        weights_pair = [extract_weights(self.target) for network in networks]
        yield from zip(*weights_pair)

    def calculate_total_distance(self) -> tuple[float, ...]:
        pairs = self.iterate_compared_layers()
        return tuple(self.calculate_distance(*pair) for pair in pairs)

    @classmethod
    def calculate_distance(cls, values: torch.Tensor, other: torch.Tensor) -> float:
        return l1_loss(values, other).item()

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
