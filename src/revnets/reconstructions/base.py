from dataclasses import dataclass

import torch

from revnets.evaluations.weights.standardize.standardize import get_layers

from ..context import context
from ..networks.models.base import Model
from ..networks.train import Network
from ..utils import NamedClass


@dataclass
class Reconstructor(NamedClass):
    network: Network
    reconstruction: Model = None
    downscale_factor: float = context.config.weight_variance_downscale_factor

    def __post_init__(self) -> None:
        self.original: Model = self.network.create_trained_model()
        self.reconstruction = self.initialize_reconstruction()

    def reconstruct(self):
        self.reconstruct_weights()
        return self.reconstruction

    def initialize_reconstruction(self):
        reconstruction = self.network.create_initialized_model()
        if self.downscale_factor is not None:
            self.scale_weights(reconstruction)
        self.set_biases(reconstruction)
        return reconstruction

    def scale_weights(self, model) -> None:
        layers = get_layers(model)
        for layer in layers:
            layer.weight.data /= self.downscale_factor

    @classmethod
    def set_biases(cls, model) -> None:
        layers = get_layers(model)
        for layer in layers:
            bias = torch.zeros_like(layer.bias, dtype=layer.bias.dtype)
            layer.bias = torch.nn.Parameter(bias)

    def reconstruct_weights(self) -> None:
        pass

    @classmethod
    def get_base_name(cls):
        return Reconstructor.__module__
