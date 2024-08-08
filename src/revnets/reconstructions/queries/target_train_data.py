from dataclasses import dataclass

import torch

from .iterative import difficult_train_inputs


@dataclass
class Reconstructor(difficult_train_inputs.Reconstructor):
    def create_queries(self, num_samples: int) -> torch.Tensor:  # noqa: ARG002
        inputs = self.pipeline.load_all_train_inputs()
        recombined_inputs = self.recombine(inputs)
        return (
            recombined_inputs + self.create_random_inputs(recombined_inputs.shape) / 100
        )
