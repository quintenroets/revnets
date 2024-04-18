from dataclasses import dataclass
from typing import cast

import torch

from .scale_isomorphism import ScaleIsomorphism
from .weights import Weights, feedforward


@dataclass
class Layer:
    weights: Weights
    scale_isomorphism: ScaleIsomorphism | None


@dataclass
class InternalLayer(Layer):
    next: Weights
    standardized_scale: float = 1

    def standardize_scale(self) -> None:
        """
        Standardize by multiplying incoming weights and biases by scale and outgoing
        weights with the inverse scale.
        """
        if self.scale_isomorphism is not None:
            self._standardize_scale()

    def _standardize_scale(self) -> None:
        scale_factors = self.calculate_scale_factors()
        # only feedforward layers have scale isomorphism
        weights = cast(feedforward.Weights, self.weights)
        weights.rescale_outgoing(1 / scale_factors)

        # we assume that an rnn layer is never preceded by a feedforward layer
        next = cast(feedforward.Weights, self.next)
        next.rescale_incoming(scale_factors)

    def calculate_scale_factors(self) -> torch.Tensor:
        # only feedforward layers have scale isomorphism
        weights = cast(feedforward.Weights, self.weights)
        isomorphism = cast(ScaleIsomorphism, self.scale_isomorphism)
        return weights.calculate_outgoing_scales(isomorphism) / self.standardized_scale

    def standardize_order(self) -> None:
        sort_indices = self.weights.calculate_outgoing_sort_order()
        self.weights.permute_outgoing(sort_indices)
        self.next.permute_incoming(sort_indices)
