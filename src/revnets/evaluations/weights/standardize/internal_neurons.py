from dataclasses import dataclass

import torch

from . import order


@dataclass
class InternalNeurons:
    incoming: torch.nn.Module
    outgoing: torch.nn.Module

    def standardize_scale(self, standardized_scale: float = 1) -> None:
        """
        Standardize by multiplying incoming weights and biases by scale and outgoing
        weights with the inverse scale.
        """
        scale_factors = calculate_incoming_weight_scales(self.incoming)
        scale_factors /= standardized_scale
        rescale_incoming_weights(self.incoming, 1 / scale_factors)
        rescale_outgoing_weights(self.outgoing, scale_factors)


def calculate_incoming_weight_scales(layer: torch.nn.Module) -> torch.Tensor:
    weights = order.get_layer_weights(layer)
    tanh = False
    if tanh:
        # tanh has sign invariance instead of scale invariance
        scales = torch.sign(weights.sum(dim=1))
    else:
        scales = weights.norm(dim=1, p=2)
    return scales


def rescale_incoming_weights(layer, scales) -> None:
    for param in layer.parameters():
        multiplier = scales if len(param.data.shape) == 1 else scales.reshape(-1, 1)
        param.data *= multiplier


def rescale_outgoing_weights(layer, scales) -> None:
    for param in layer.parameters():
        if len(param.shape) == 2:
            param.data *= scales
