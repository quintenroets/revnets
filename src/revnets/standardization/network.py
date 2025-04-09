from dataclasses import dataclass
from functools import cached_property
from typing import TypeVar, cast

from torch import nn

from .layer import InternalLayer
from .scale_isomorphism import ScaleIsomorphism
from .utils import extract_internal_layers
from .weights import feedforward

T = TypeVar("T")


@dataclass
class Standardizer:
    model: nn.Module
    optimize_mae: bool = False

    def run(self) -> None:
        """
        Convert network to the standard form of its isomorphism group.
        """
        self.standardize_scale()
        for layer in self.internal_layers:
            layer.standardize_order()
        if self.optimize_mae:
            self.distribute_total_output_scale()

    def standardize_scale(self) -> None:
        for layer in self.internal_layers:
            layer.standardize_scale()

    def distribute_total_output_scale(self) -> None:
        """
        Distribute total output scale equally across all layers.
        """
        last_layer = self.internal_layers[-1]
        can_distribute = last_layer.scale_isomorphism is ScaleIsomorphism.norm
        if can_distribute and self.layers_with_norm_isomorphism:
            desired_scale = self.calculate_average_scale_per_layer()
            for layer in self.layers_with_norm_isomorphism:
                layer.standardized_scale = desired_scale
                layer.standardize_scale()

    def calculate_average_scale_per_layer(self) -> float:
        next_ = cast("feedforward.Weights", self.internal_layers[-1].next)
        scales = next_.calculate_outgoing_norms()
        scale = sum(scales) / len(scales)
        num_internal_layers = len(self.layers_with_norm_isomorphism)
        average_scale = scale ** (1 / num_internal_layers)
        return cast("float", average_scale)

    @cached_property
    def internal_layers(self) -> list[InternalLayer]:
        return list(extract_internal_layers(self.model))

    @cached_property
    def layers_with_norm_isomorphism(self) -> list[InternalLayer]:
        return [
            layer
            for layer in self.internal_layers
            if layer.scale_isomorphism is ScaleIsomorphism.norm
        ]
