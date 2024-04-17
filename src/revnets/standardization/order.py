from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import nn

from .parameters import Parameters
from .internal_connection import InternalConnection


@dataclass
class RNNLayerParameters:
    input_to_hidden: Parameters
    hidden_to_hidden: Parameters

    def permute_incoming(self, sort_indices: torch.Tensor) -> None:
        self.input_to_hidden.permute_incoming(sort_indices)

    def permute_outgoing(self, sort_indices: torch.Tensor) -> None:
        self.input_to_hidden.permute_outgoing(sort_indices)
        self.hidden_to_hidden.permute_incoming(sort_indices)
        self.hidden_to_hidden.permute_outgoing(sort_indices)

    def calculate_outgoing_sort_order(self) -> torch.Tensor:
        return self.input_to_hidden.calculate_outgoing_sort_order()


@dataclass
class Standardizer:
    connection: InternalConnection

    def run(self) -> None:
        if isinstance(self.connection.input, nn.RNN):
            self.run_on_rnn()
        else:
            self._run()

    def run_on_rnn(self) -> None:
        rnn_layers = list(self.generate_rnn_layer_parameters(self.connection.input))
        sort_indices = None
        for input_layer in rnn_layers:
            if sort_indices is not None:
                input_layer.permute_incoming(sort_indices)
            sort_indices = input_layer.calculate_outgoing_sort_order()
            input_layer.permute_outgoing(sort_indices)
        self.connection.output_parameters.permute_incoming(sort_indices)

    def _run(self) -> None:
        sort_indices = self.connection.input_parameters.calculate_outgoing_sort_order()
        self.connection.input_parameters.permute_outgoing(sort_indices)
        self.connection.output_parameters.permute_incoming(sort_indices)

    def generate_rnn_layer_parameters(
        self, rnn: nn.RNN
    ) -> Iterator[RNNLayerParameters]:
        for layer in range(rnn.num_layers):
            input_to_hidden = self.extract_rrn_layer_parameters(rnn, layer, "i")
            hidden_to_hidden = self.extract_rrn_layer_parameters(rnn, layer, "h")
            yield RNNLayerParameters(input_to_hidden, hidden_to_hidden)

    def extract_rrn_layer_parameters(
        self, rnn: nn.RNN, layer: int, input_type: str
    ) -> Iterator[Parameters]:
        weights = getattr(rnn, f"weight_{input_type}h_l{layer}")
        bias = getattr(rnn, f"bias_{input_type}h_l{layer}")
        return Parameters(weights, bias)
