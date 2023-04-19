from dataclasses import dataclass

import torch

from .. import base
from . import standardize


@dataclass
class Evaluator(base.Evaluator):
    use_align: bool = True

    def evaluate(self):
        return self.calculate_distance() if self.standardize_networks() else None

    def standardize_networks(self):
        standardized = self.same_architecture()
        if standardized:
            if self.use_align:
                standardize.align.align(self.original, self.reconstruction)
            else:
                for model in (self.original, self.reconstruction):
                    standardize.standardize.standardize(model)
        return standardized

    def same_architecture(self):
        return all(
            original.shape == reconstruction.shape
            for original, reconstruction in self.iterate_compared_layers()
        )

    def calculate_distance(self):
        total_size = sum(
            weights.numel() for weights in self.original.state_dict().values()
        )
        total_distance = sum(
            self.calculate_weights_distance(original, reconstruction)
            for original, reconstruction in self.iterate_compared_layers()
        )
        distance = total_distance / total_size
        return distance

    @classmethod
    def calculate_weights_distance(cls, original, reconstruction):
        distance = torch.nn.functional.mse_loss(
            original, reconstruction, reduction="sum"
        )
        return distance.item()

    def iterate_compared_layers(self):
        yield from zip(
            self.original.state_dict().values(),
            self.reconstruction.state_dict().values(),
        )
