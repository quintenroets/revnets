from dataclasses import dataclass, field
from typing import cast

from torch.nn import Sequential

from revnets.pipelines import Pipeline
from revnets.standardization import extract_layers

from ..context import context
from ..utils import NamedClass


@dataclass
class Reconstructor(NamedClass):
    pipeline: Pipeline
    downscale_factor: float | None = field(
        default_factory=lambda: context.config.weight_variance_downscale_factor
    )
    reconstruction: Sequential = field(init=False)

    def __post_init__(self) -> None:
        self.reconstruction = self.pipeline.create_initialized_network()

    def create_reconstruction(self) -> Sequential:
        if self.downscale_factor is not None:
            self.scale_weights()  # pragma: nocover
        if context.config.start_reconstruction_with_zero_biases:
            self.set_biases_to_zero()  # pragma: nocover
        self.reconstruct_weights()
        return self.reconstruction

    def scale_weights(self) -> None:  # pragma: nocover
        layers = extract_layers(self.reconstruction)
        for layer in layers:
            downscale_factor = cast(float, self.downscale_factor)
            layer.weights.scale_down(downscale_factor)

    def set_biases_to_zero(self) -> None:  # pragma: nocover
        layers = extract_layers(self.reconstruction)
        for layer in layers:
            layer.weights.set_biases_to_zero()

    def reconstruct_weights(self) -> None:
        raise NotImplementedError  # pragma: nocover

    @classmethod
    def get_base_name(cls) -> str:
        return Reconstructor.__module__
