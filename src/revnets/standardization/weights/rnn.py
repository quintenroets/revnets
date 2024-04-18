from dataclasses import dataclass

import torch
from torch import nn

from . import base, feedforward


@dataclass
class Weights(base.Weights):
    input_to_hidden: feedforward.Weights
    hidden_to_hidden: feedforward.Weights

    def permute_incoming(self, sort_indices: torch.Tensor) -> None:
        self.input_to_hidden.permute_incoming(sort_indices)

    def permute_outgoing(self, sort_indices: torch.Tensor) -> None:
        self.input_to_hidden.permute_outgoing(sort_indices)
        self.hidden_to_hidden.permute_incoming(sort_indices)
        self.hidden_to_hidden.permute_outgoing(sort_indices)

    def calculate_outgoing_sort_order(self) -> torch.Tensor:
        return self.input_to_hidden.calculate_outgoing_sort_order()

    @property
    def weights(self) -> torch.Tensor:
        return self.input_to_hidden.weights

    def scale_down(self, scale: float) -> None:
        self.input_to_hidden.scale_down(scale)
        self.hidden_to_hidden.scale_down(scale)

    def set_biases_to_zero(self) -> None:
        self.input_to_hidden.set_biases_to_zero()
        self.hidden_to_hidden.set_biases_to_zero()


def extract_weights(rnn: nn.RNN, layer: int) -> Weights:
    input_to_hidden = extract_feedforward_weights(rnn, layer, "i")
    hidden_to_hidden = extract_feedforward_weights(rnn, layer, "h")
    return Weights(input_to_hidden, hidden_to_hidden)


def extract_feedforward_weights(
    rnn: nn.RNN, layer: int, input_type: str
) -> feedforward.Weights:
    weights = getattr(rnn, f"weight_{input_type}h_l{layer}")
    bias = getattr(rnn, f"bias_{input_type}h_l{layer}")
    return feedforward.Weights(weights, bias)
