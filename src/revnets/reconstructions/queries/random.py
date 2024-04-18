from collections.abc import Sequence
from dataclasses import dataclass

import torch

from revnets.context import context

from . import base


@dataclass
class Reconstructor(base.Reconstructor):
    def create_queries(self, num_samples: int) -> torch.Tensor:
        shape = num_samples, *self.input_shape
        return self.create_random_inputs(shape)

    @property
    def input_shape(self) -> tuple[int, ...]:
        dataset = self.pipeline.load_prepared_data()
        return dataset.input_shape

    def create_random_inputs(self, shape: Sequence[int]) -> torch.Tensor:
        inputs = self.pipeline.load_all_train_inputs()
        mean = inputs.mean().item()
        std = inputs.std().item()
        # same mean and std as training data (should be 0 and 1)
        return torch.randn(shape, dtype=context.dtype) * std + mean
