from dataclasses import dataclass

import torch

from revnets.evaluations.weights.standardize.standardize import get_layers

from ..networks.base import Network
from ..networks.models.base import Model
from ..utils import NamedClass, config


@dataclass
class Reconstructor(NamedClass):
    network: Network
    reconstruction: Model = None
    downscale_factor: float = config.weight_variance_downscale_factor

    def __post_init__(self):
        self.original: Model = self.network.trained_network
        self.reconstruction = self.initialize_reconstruction()

    def reconstruct(self):
        self.reconstruct_weights()
        return self.reconstruction

    def initialize_reconstruction(self):
        reconstruction = self.network.get_architecture()
        if self.downscale_factor is not None:
            self.scale_weights(reconstruction)
        self.set_biases(reconstruction)
        return reconstruction

    def scale_weights(self, model):
        layers = get_layers(model)
        for layer in layers:
            layer.weight.data /= self.downscale_factor

    @classmethod
    def set_biases(cls, model):
        layers = get_layers(model)
        for layer in layers:
            bias = torch.zeros_like(layer.bias, dtype=layer.bias.dtype)
            layer.bias = torch.nn.Parameter(bias)

    def reconstruct_weights(self):
        pass

    @classmethod
    def get_base_name(cls):
        return Reconstructor.__module__
