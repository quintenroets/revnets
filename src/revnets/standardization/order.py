from dataclasses import dataclass

import torch
from torch import nn

from revnets.models import InternalNeurons

from .utils import extract_parameters, extract_weights


@dataclass
class Standardizer:
    neurons: InternalNeurons

    def run(self) -> None:
        sort_indices = calculate_sort_order(self.neurons.incoming)
        permute_output_weights(self.neurons.incoming, sort_indices)
        permute_input_weights(self.neurons.outgoing, sort_indices)


def calculate_sort_order(layer: nn.Module) -> torch.Tensor:
    weights = extract_weights(layer)
    p = 1  # use l1-norm because l2-norm is already standardized
    sort_values = weights.norm(dim=1, p=p)
    return torch.sort(sort_values)[1]


def permute_input_weights(layer: nn.Module, sort_indices: torch.Tensor) -> None:
    parameters = extract_parameters(layer)
    number_of_outputs = len(parameters.weight)
    flat_connections = parameters.weight.data.reshape(number_of_outputs, -1)
    permuted_connections = flat_connections[:, sort_indices]
    parameters.weight.data = permuted_connections.reshape(parameters.weight.shape)


def permute_output_weights(layer: nn.Module, sort_indices: torch.Tensor) -> None:
    parameters = extract_parameters(layer)
    parameters.weight.data = parameters.weight.data[sort_indices, :]
    if parameters.bias is not None:
        parameters.bias.data = parameters.bias.data[sort_indices]
