from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from kneed import KneeLocator
from torch.utils.data import DataLoader

from revnets.utils.data import compute_targets

from . import base


@dataclass
class Reconstructor(base.Reconstructor):
    noise_factor: float = 1 / 100

    def create_difficult_samples(self) -> torch.Tensor:
        difficult_inputs = self.extract_difficult_inputs()
        recombined_inputs = self.recombine(difficult_inputs)
        noise = self.create_random_inputs(recombined_inputs.shape)
        return recombined_inputs + self.noise_factor * noise

    def recombine(self, inputs: torch.Tensor) -> torch.Tensor:
        new_samples_shape = self.num_samples, inputs.shape[-1]
        n_inputs = len(inputs)
        new_samples = np.random.choice(range(n_inputs), size=new_samples_shape)
        # each feature value in a new sample corresponds with a feature value
        # in the corresponding feature of one of the inputs
        return inputs[new_samples, np.arange(new_samples.shape[1])]

    def extract_difficult_inputs(self) -> torch.Tensor:
        data = self.pipeline.load_prepared_data()
        batch_size = len(data.train)  # type: ignore[arg-type]
        dataloader = DataLoader(data.train, batch_size=batch_size, shuffle=False)
        inputs = next(iter(dataloader))[0]
        outputs = compute_targets(inputs, self.reconstruction)
        targets = compute_targets(inputs, self.pipeline.target)
        high_loss_indices = self.extract_high_loss_indices(outputs, targets)
        difficult_inputs = inputs[high_loss_indices]
        return cast(torch.Tensor, difficult_inputs)

    def extract_high_loss_indices(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        sorted_losses, original_indices = self.calculate_sorted_losses(outputs, targets)
        elbow = self.calculate_elbow(sorted_losses)
        return original_indices[:elbow]

    @classmethod
    def calculate_sorted_losses(
        cls, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.return_types.sort:
        losses = torch.nn.functional.l1_loss(outputs, targets, reduction="none")
        losses = losses.mean(dim=1)
        return torch.sort(losses, descending=True)

    @classmethod
    def calculate_elbow(cls, values: torch.Tensor) -> int:
        elbow_range = range(len(values))
        elbow_result = KneeLocator(
            elbow_range, values, curve="convex", direction="decreasing"
        )
        return cast(int, elbow_result.elbow)
