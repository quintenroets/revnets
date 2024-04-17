from dataclasses import dataclass

import torch
from torch.nn import Module, Parameter


@dataclass
class Parameters:
    weight: Parameter
    bias: Parameter | None

    @property
    def number_of_outputs(self) -> int:
        return self.weight.shape[0]

    @property
    def weights(self) -> torch.Tensor:
        with torch.no_grad():
            return self.extract_weights()

    def extract_weights(self) -> torch.Tensor:
        # flatten incoming weights for each out layer output (Conv, ..)
        shape = self.number_of_outputs, -1
        weights = self.weight.reshape(shape)
        if self.bias is not None:
            bias = self.bias.reshape(shape)
            combined_weights = (weights, bias)
            weights = torch.hstack(combined_weights)
        return weights

    def permute_incoming(self, sort_indices: torch.Tensor) -> None:
        # take into account that flatten layers cause outgoing weights with altered shapes
        shape = self.weight.data.shape[0], sort_indices.shape[0], -1
        data = self.weight.data.view(shape)
        data = torch.index_select(data, 1, sort_indices)
        self.weight.data = data.reshape(self.weight.shape)

    def permute_outgoing(self, sort_indices: torch.Tensor) -> None:
        self.weight.data = self.weight.data[sort_indices]
        if self.bias is not None:
            self.bias.data = self.bias.data[sort_indices]

    def calculate_outgoing_sort_order(self) -> torch.Tensor:
        p = 1  # use l1-norm because l2-norm is already standardized
        sort_values = self.weights.norm(dim=1, p=p)
        return torch.sort(sort_values)[1]


def extract_parameters(layer: Module) -> Parameters:
    parameters = layer.parameters()
    weight = next(parameters)
    bias = next(parameters, None)
    return Parameters(weight, bias)
