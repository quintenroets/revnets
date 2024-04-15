from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch.utils.data import DataLoader

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
        train_inputs = self.extract_train_inputs()
        mean = train_inputs.mean().item()
        std = train_inputs.std().item()
        # same mean and std as training data (should be 0 and 1)
        return torch.randn(shape, dtype=context.dtype) * std + mean

    def extract_train_inputs(self) -> torch.Tensor:
        data = self.pipeline.load_prepared_data()
        batch_size = len(data.train_validation)  # type: ignore[arg-type]
        dataloader = DataLoader(data.train_validation, batch_size, shuffle=False)
        batch = next(iter(dataloader))
        return cast(torch.Tensor, batch[0])
