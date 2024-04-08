from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from kneed import KneeLocator

from revnets.data import output_supervision

from . import base


@dataclass
class Reconstructor(base.Reconstructor):
    noise_factor: float = 1 / 100

    def get_difficult_inputs(self) -> torch.Tensor:
        difficult_inputs = self.extract_difficult_inputs()
        recombined_inputs = self.recombine(difficult_inputs)
        noise = self.data.generate_random_inputs(recombined_inputs.shape)
        new_difficult_inputs = recombined_inputs + self.noise_factor * noise
        return new_difficult_inputs

    def recombine(self, inputs: torch.Tensor) -> torch.Tensor:
        new_samples_shape = self.num_samples, inputs.shape[-1]
        n_inputs = len(inputs)
        new_samples = np.random.choice(range(n_inputs), size=new_samples_shape)
        # each feature value in a new sample corresponds with a feature value
        # in the corresponding feature of one of the inputs
        return inputs[new_samples, np.arange(new_samples.shape[1])]

    def extract_difficult_inputs(self) -> torch.Tensor:
        split = output_supervision.Split.train
        inputs = self.data.get_all_inputs(split)
        targets = self.data.get_all_targets(split)

        outputs_dataset = output_supervision.Dataset(
            original_dataset=self.data, target_network=self.network
        )
        outputs = outputs_dataset.add_output_supervision(split).tensors[1]

        high_loss_indices = self.get_high_loss_indices(outputs, targets)
        high_loss_inputs = inputs[high_loss_indices]
        return high_loss_inputs

    def get_high_loss_indices(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        sorted_losses, original_indices = self.calculate_sorted_losses(outputs, targets)
        elbow = self.get_elbow(sorted_losses)
        return original_indices[:elbow]

    @classmethod
    def calculate_sorted_losses(
        cls, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.return_types.sort:
        losses = torch.nn.functional.l1_loss(outputs, targets, reduction="none")
        losses = losses.mean(dim=1)
        return torch.sort(losses, descending=True)

    @classmethod
    def get_elbow(cls, values: torch.Tensor) -> int:
        elbow_range = range(len(values))
        elbow_result = KneeLocator(
            elbow_range, values, curve="convex", direction="decreasing"
        )
        return cast(int, elbow_result.elbow)
