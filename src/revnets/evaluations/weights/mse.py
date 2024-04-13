from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import torch

from revnets.standardization import Standardizer, align

from ...context import context
from .. import base


@dataclass
class Evaluator(base.Evaluator):
    def evaluate(self) -> float | tuple[float, ...] | None:
        return self.calculate_distance() if self.standardize_networks() else None

    def standardize_networks(self) -> bool:
        standardized = self.has_same_architecture()
        if standardized:
            if context.config.evaluation.use_align:
                align(self.target, self.reconstruction)  # pragma: nocover
            else:
                for network in (self.target, self.reconstruction):
                    Standardizer(network).run()
        return standardized

    def has_same_architecture(self) -> bool:
        return all(
            original.shape == reconstruction.shape
            for original, reconstruction in self.iterate_compared_layers()
        )

    def calculate_distance(self) -> float | tuple[float, ...]:
        layer_weights = self.target.state_dict().values()
        total_size = sum(weights.numel() for weights in layer_weights)
        total_distance = sum(
            self.calculate_weights_distance(original, reconstruction)
            for original, reconstruction in self.iterate_compared_layers()
        )
        distance = total_distance / total_size
        return cast(float, distance)

    @classmethod
    def calculate_weights_distance(
        cls, original: torch.Tensor, reconstruction: torch.Tensor
    ) -> float:
        distance = torch.nn.functional.mse_loss(
            original, reconstruction, reduction="sum"
        )
        return distance.item()

    def iterate_compared_layers(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        yield from zip(
            self.target.state_dict().values(),
            self.reconstruction.state_dict().values(),
        )
