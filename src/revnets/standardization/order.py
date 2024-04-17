from dataclasses import dataclass

import torch

from .internal_connection import InternalConnection, Parameters


@dataclass
class Standardizer:
    connection: InternalConnection

    def run(self) -> None:
        sort_indices = calculate_outgoing_sort_order(self.connection.input_weights)
        permute_outgoing(self.connection.input_parameters, sort_indices)
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
