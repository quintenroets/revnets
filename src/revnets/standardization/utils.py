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


def extract_parameters(layer: Module) -> Parameters:
    parameters = layer.parameters()
    weight = next(parameters)
    bias = next(parameters, None)
    return Parameters(weight, bias)


def extract_weights(layer: Module) -> torch.Tensor:
    with torch.no_grad():
        return _extract_weights(layer)


def _extract_weights(layer: Module) -> torch.Tensor:
    parameters = extract_parameters(layer)
    # flatten incoming weights for each out layer output (Conv, ..)
    shape = parameters.number_of_outputs, -1
    weights = parameters.weight.reshape(shape)
    if parameters.bias is not None:
        bias = parameters.bias.reshape(shape)
        combined_weights = (weights, bias)
        weights = torch.hstack(combined_weights)
    return weights