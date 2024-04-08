from dataclasses import dataclass
from typing import cast

import torch
from torch.nn import Module

from revnets.models import InternalNeurons

from .utils import extract_linear_layer_weights


@dataclass
class Standardizer:
    neurons: InternalNeurons

    def run(self) -> None:
        """
        Standardize by multiplying incoming weights and biases by scale and outgoing
        weights with the inverse scale.
        """
        if self.neurons.has_scale_isomorphism:
            self.standardize_scale()

    def standardize_scale(self) -> None:
        scale_factors = self.calculate_scale_factors(self.neurons.incoming)
        scale_factors /= self.neurons.standardized_scale
        rescale_incoming_weights(self.neurons.incoming, 1 / scale_factors)
        rescale_outgoing_weights(self.neurons.outgoing, scale_factors)

    def calculate_scale_factors(self, layer: Module) -> torch.Tensor:
        weights = extract_linear_layer_weights(layer)
        scale_factors = (
            torch.sign(weights.sum(dim=1))
            if self.neurons.has_sign_isomorphism
            else weights.norm(dim=1, p=2)
        )
        return cast(torch.Tensor, scale_factors)


def rescale_incoming_weights(layer: Module, scales: torch.Tensor) -> None:
    for param in layer.parameters():
        multiplier = scales if len(param.data.shape) == 1 else scales.reshape(-1, 1)
        param.data *= multiplier


def rescale_outgoing_weights(layer: Module, scales: torch.Tensor) -> None:
    for param in layer.parameters():
        if len(param.shape) == 2:
            param.data *= scales
