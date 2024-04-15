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
        permute_incoming_weights(self.neurons.incoming, sort_indices)
        permute_outgoing_weights(self.neurons.outgoing, sort_indices)


def calculate_sort_order(layer: nn.Module) -> torch.Tensor:
    weights = extract_weights(layer)
    p = 1  # use l1-norm because l2-norm is already standardized
    sort_values = weights.norm(dim=1, p=p)
    return torch.sort(sort_values)[1]


def permute_incoming_weights(layer: nn.Module, sort_indices: torch.Tensor) -> None:
    parameters = extract_parameters(layer)
    parameters.weight.data = parameters.weight.data[sort_indices]
    if parameters.bias is not None:
        parameters.bias.data = parameters.bias.data[sort_indices]


def permute_outgoing_weights(layer: nn.Module, sort_indices: torch.Tensor) -> None:
    parameters = extract_parameters(layer)
    # take into account that flatten layers cause outgoing weights with altered shapes
    shape = parameters.weight.data.shape[0], sort_indices.shape[0], -1
    data = parameters.weight.data.view(shape)
    data = torch.index_select(data, 1, sort_indices)
    parameters.weight.data = data.reshape(parameters.weight.shape)
