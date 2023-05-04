from dataclasses import dataclass

from revnets.evaluations.weights.standardize.standardize import get_layers

from . import random_inputs


@dataclass
class Reconstructor(random_inputs.Reconstructor):
    scale_factor: float = 10

    def initialize_reconstruction(self):
        reconstruction = super().initialize_reconstruction()
        layers = get_layers(reconstruction)
        for layer in layers:
            layer.weight.data /= self.scale_factor
        return reconstruction
