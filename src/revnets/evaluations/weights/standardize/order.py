from dataclasses import dataclass

import torch
from torch import nn

from revnets.models import InternalNeurons

from .utils import extract_linear_layer_weights


@dataclass
class Standardizer:
    neurons: InternalNeurons

    def run(self) -> None:
        sort_indices = calculate_sort_order(self.neurons.incoming)
        permute_output_weights(self.neurons.incoming, sort_indices)
        permute_input_weights(self.neurons.outgoing, sort_indices)


def calculate_sort_order(layer: nn.Module) -> torch.Tensor:
    weights = extract_linear_layer_weights(layer)
    p = 1  # use l1-norm because l2-norm is already standardized
    total_output_weights = weights.norm(dim=1, p=p)
    return torch.sort(total_output_weights)[1]


def permute_input_weights(layer: nn.Module, sort_indices: torch.Tensor) -> None:
    length = len(sort_indices)
    for param in layer.parameters():
        shape = param.data.shape
        if len(shape) == 2 and shape[1] == length:
            param.data = param.data[:, sort_indices]


def permute_output_weights(layer: nn.Module, sort_indices: torch.Tensor) -> None:
    length = len(sort_indices)
    for param in layer.parameters():
        shape = param.shape
        dims = len(shape)
        if dims in (1, 2) and shape[0] == length:
            param.data = (
                param.data[sort_indices] if dims == 1 else param.data[sort_indices, :]
            )
