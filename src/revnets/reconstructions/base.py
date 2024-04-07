from dataclasses import dataclass, field

import torch
from torch.nn import Sequential

from revnets.evaluations.weights.standardize.standardize import generate_layers
from revnets.pipelines import Pipeline

from ..context import context
from ..utils import NamedClass


@dataclass
class Reconstructor(NamedClass):
    pipeline: Pipeline
    downscale_factor: float | None = context.config.weight_variance_downscale_factor
    reconstruction: Sequential = field(init=False)

    def __post_init__(self) -> None:
        self.reconstruction = self.pipeline.create_initialized_network()

    def create_reconstruction(self) -> Sequential:
        if self.downscale_factor is not None:
            self.scale_weights()
        if context.config.start_reconstruction_with_zero_biases:
            self.set_biases()
        self.reconstruct_weights()
        return self.reconstruction

    def scale_weights(self) -> None:
        layers = generate_layers(self.reconstruction)
        for layer in layers:
            layer.weight.data /= self.downscale_factor

    def set_biases(self) -> None:
        layers = generate_layers(self.reconstruction)
        for layer in layers:
            bias = torch.zeros_like(layer.bias, dtype=layer.bias.dtype)
            layer.bias = torch.nn.Parameter(bias)

    def reconstruct_weights(self) -> None:
        raise NotImplementedError

    @classmethod
    def get_base_name(cls) -> str:
        return Reconstructor.__module__
