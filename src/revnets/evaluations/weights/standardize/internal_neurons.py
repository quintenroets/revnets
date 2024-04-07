from dataclasses import dataclass

import torch
from torch.nn import LeakyReLU, ReLU, Tanh

from . import order


@dataclass
class InternalNeurons:
    incoming: torch.nn.Module
    activation: torch.nn.Module
    outgoing: torch.nn.Module
    standardized_scale: float = 1

    @property
    def has_norm_isomorphism(self) -> bool:
        activations = ReLU, LeakyReLU
        return isinstance(self.activation, activations)

    @property
    def has_sign_isomorphism(self) -> bool:
        return isinstance(self.activation, Tanh)

    @property
    def has_scale_isomorphism(self) -> bool:
        return self.has_norm_isomorphism or self.has_sign_isomorphism

    def standardize_scale(self) -> None:
        """
        Standardize by multiplying incoming weights and biases by scale and outgoing
        weights with the inverse scale.
        """
        if self.has_scale_isomorphism:
            self._standardize_scale()

    def _standardize_scale(self) -> None:
        scale_factors = self.calculate_scale_factors(self.incoming)
        scale_factors /= self.standardized_scale
        rescale_incoming_weights(self.incoming, 1 / scale_factors)
        rescale_outgoing_weights(self.outgoing, scale_factors)

    def calculate_scale_factors(self, layer: torch.nn.Module) -> torch.Tensor:
        weights = order.get_layer_weights(layer)
        return (
            torch.sign(weights.sum(dim=1))
            if self.has_sign_isomorphism
            else weights.norm(dim=1, p=2)
        )


def rescale_incoming_weights(layer, scales) -> None:
    for param in layer.parameters():
        multiplier = scales if len(param.data.shape) == 1 else scales.reshape(-1, 1)
        param.data *= multiplier


def rescale_outgoing_weights(layer, scales) -> None:
    for param in layer.parameters():
        if len(param.shape) == 2:
            param.data *= scales
