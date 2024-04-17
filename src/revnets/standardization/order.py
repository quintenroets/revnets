from dataclasses import dataclass

import torch
from torch import nn

from .internal_connection import InternalConnection, Parameters
from .utils import extract_weights


@dataclass
class Standardizer:
    connection: InternalConnection

    def run(self) -> None:
        if isinstance(self.connection.input, nn.RNN):
            self.run_on_rnn()
        else:
            self._run()

    def _run(self) -> None:
        sort_indices = calculate_outgoing_sort_order(self.connection.input_weights)
        permute_outgoing(self.connection.input_parameters, sort_indices)
        permute_incoming(self.connection.output_parameters, sort_indices)

    def run_on_rnn(self) -> None:
        def _extract_parameters(rnn_layer: int, t: str) -> Parameters:
            weights = getattr(layer, f"weight_{t}h_l{rnn_layer}")
            bias = getattr(layer, f"bias_{t}h_l{rnn_layer}")
            return Parameters(weights, bias)

        layer = self.connection.input
        sort_indices = None
        for rnn_layer in range(layer.num_layers):
            input_to_hidden = _extract_parameters(rnn_layer, "i")
            hidden_to_hidden = _extract_parameters(rnn_layer, "h")
            if sort_indices is not None:
                permute_incoming(input_to_hidden, sort_indices)
            sort_indices = calculate_outgoing_sort_order(
                extract_weights(input_to_hidden)
            )
            permute_outgoing(input_to_hidden, sort_indices)
            permute_incoming(hidden_to_hidden, sort_indices)
            permute_outgoing(hidden_to_hidden, sort_indices)
        permute_incoming(self.connection.output_parameters, sort_indices)


def calculate_outgoing_sort_order(weights: torch.Tensor) -> torch.Tensor:
    p = 1  # use l1-norm because l2-norm is already standardized
    sort_values = weights.norm(dim=1, p=p)
    return torch.sort(sort_values)[1]


def permute_outgoing(parameters: Parameters, sort_indices: torch.Tensor) -> None:
    parameters.weight.data = parameters.weight.data[sort_indices]
    if parameters.bias is not None:
        parameters.bias.data = parameters.bias.data[sort_indices]


def permute_incoming(parameters: Parameters, sort_indices: torch.Tensor) -> None:
    # take into account that flatten layers cause outgoing weights with altered shapes
    shape = parameters.weight.data.shape[0], sort_indices.shape[0], -1
    data = parameters.weight.data.view(shape)
    data = torch.index_select(data, 1, sort_indices)
    parameters.weight.data = data.reshape(parameters.weight.shape)
