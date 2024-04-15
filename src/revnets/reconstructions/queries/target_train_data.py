from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from .iterative import difficult_train_inputs


@dataclass
class Reconstructor(difficult_train_inputs.Reconstructor):
    def create_queries(self, num_samples: int) -> torch.Tensor:
        data = self.pipeline.load_prepared_data().train_validation
        batch_size = len(data)  # type: ignore[arg-type]
        dataloader = DataLoader(data, batch_size, shuffle=False)
        inputs, _ = next(iter(dataloader))
        recombined_inputs = self.recombine(inputs)
        return (
            recombined_inputs + self.create_random_inputs(recombined_inputs.shape) / 100
        )
