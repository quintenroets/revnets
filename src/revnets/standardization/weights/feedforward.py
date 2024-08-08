from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from revnets.standardization.scale_isomorphism import ScaleIsomorphism

from . import base


@dataclass
class Weights(base.Weights):
    weight: nn.Parameter
    bias: nn.Parameter | None

    def permute_incoming(self, sort_indices: torch.Tensor) -> None:
        data = self.extract_weight_data(sort_indices.shape)
        data = torch.index_select(data, 1, sort_indices)
        self.set_weight_data(data)

    def permute_outgoing(self, sort_indices: torch.Tensor) -> None:
        self.weight.data = self.weight.data[sort_indices]
        if self.bias is not None:
            self.bias.data = self.bias.data[sort_indices]

    def calculate_outgoing_sort_order(self) -> torch.Tensor:
        p = 1  # use l1-norm because l2-norm is already standardized
        sort_values = self.weights.norm(dim=1, p=p)
        return torch.sort(sort_values)[1]

    def calculate_outgoing_scales(self, isomorphism: ScaleIsomorphism) -> torch.Tensor:
        return (
            self.calculate_outgoing_signs()
            if isomorphism is ScaleIsomorphism.sign
            else self.calculate_outgoing_norms()
        )

    def calculate_outgoing_signs(self) -> torch.Tensor:
        return torch.sign(self.weights.sum(dim=1))

    def calculate_outgoing_norms(self) -> torch.Tensor:
        norms = self.weights.norm(dim=1, p=2)
        return cast(torch.Tensor, norms)

    def rescale_incoming(self, scales: torch.Tensor) -> None:
        data = self.extract_weight_data(scales.shape)
        rescaled_data = data * broadcast(scales, data, dimension=1)
        self.set_weight_data(rescaled_data)

    def rescale_outgoing(self, scales: torch.Tensor) -> None:
        self.weight.data *= broadcast(scales, self.weight.data)
        if self.bias is not None:
            self.bias.data *= scales

    def extract_weight_data(self, manipulation_shape: tuple[int, ...]) -> torch.Tensor:
        # take into account that flatten layers cause output weights with altered shapes
        shape = self.weight.data.shape[0], manipulation_shape[0], -1
        return self.weight.data.view(shape)

    def set_weight_data(self, data: torch.Tensor) -> None:
        self.weight.data = data.reshape(self.weight.shape)

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

    @property
    def number_of_outputs(self) -> int:
        return self.weight.shape[0]

    def scale_down(self, scale: float) -> None:
        self.weight.data /= scale

    def set_biases_to_zero(self) -> None:
        if self.bias is not None:
            values = torch.zeros_like(self.bias.data, dtype=self.bias.data.dtype)
            self.bias = torch.nn.Parameter(values)


def broadcast(
    values: torch.Tensor,
    target: torch.Tensor,
    dimension: int = 0,
) -> torch.Tensor:
    shape = [1] * target.dim()
    shape[dimension] = -1
    return values.view(*shape)


def extract_weights(module: nn.Module) -> Weights:
    parameters = module.parameters()
    weight = next(parameters)
    bias = next(parameters, None)
    return Weights(weight, bias)
