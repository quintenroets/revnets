from dataclasses import dataclass
from typing import cast

import torch

from .internal_connection import InternalConnection, Parameters


@dataclass
class Standardizer:
    connection: InternalConnection

    def run(self) -> None:
        """
        Standardize by multiplying incoming weights and biases by scale and outgoing
        weights with the inverse scale.
        """
        if self.connection.has_scale_isomorphism:
            self.standardize_scale()

    def standardize_scale(self) -> None:
        scale_factors = self.calculate_outgoing_scales(self.connection.input_weights)
        scale_factors /= self.connection.standardized_scale
        rescale_outgoing(self.connection.input_parameters, 1 / scale_factors)
        rescale_incoming(self.connection.output_parameters, scale_factors)

    def calculate_outgoing_scales(self, weights: torch.Tensor) -> torch.Tensor:
        scale_factors = (
            torch.sign(weights.sum(dim=1))
            if self.connection.has_sign_isomorphism
            else weights.norm(dim=1, p=2)
        )
        return cast(torch.Tensor, scale_factors)


def rescale_outgoing(parameters: Parameters, scales: torch.Tensor) -> None:
    parameters.weight.data *= broadcast(scales, parameters.weight.data)
    if parameters.bias is not None:
        parameters.bias.data *= scales


def broadcast(
    values: torch.Tensor, target: torch.Tensor, dimension: int = 0
) -> torch.Tensor:
    shape = [1] * target.dim()
    shape[dimension] = -1
    return values.view(*shape)


def rescale_incoming(parameters: Parameters, scales: torch.Tensor) -> None:
    # take into account that flatten layers cause outgoing weights with altered shapes
    data = parameters.weight.data
    shape = data.shape[0], scales.shape[0], -1
    data = data.view(shape)
    data *= broadcast(scales, data, dimension=1)
    parameters.weight.data = data.reshape(parameters.weight.shape)
