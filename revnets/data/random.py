from dataclasses import dataclass

import torch
from torch.utils.data import TensorDataset

from ..utils import config
from . import output_supervision
from .base import Split


@dataclass
class Dataset(output_supervision.Dataset):
    num_samples: int = 100000
    validation_ratio: float = 0.2

    def __post_init__(self):
        super().__post_init__()
        self.batch_size = config.reconstruction_batch_size

    @property
    def input_shape(self):
        train_sample = self.original_dataset.train_val_dataset[0]
        return train_sample[0].shape

    def get_dataset_shape(self, split: Split):
        num_samples = (
            self.num_samples
            if split == Split.train_val
            else int(self.num_samples * self.validation_ratio)
        )
        return num_samples, *self.input_shape

    def generate_dataset(self, split: Split):
        dataset_shape = self.get_dataset_shape(split)
        # same mean and std as training data
        inputs = torch.randn(dataset_shape, dtype=torch.float64)
        targets = self.get_targets(inputs)
        return TensorDataset(inputs, targets)

    def get_targets(self, inputs):
        inputs_dataset = TensorDataset(inputs)
        batch_size = self.original_dataset.eval_batch_size
        dataloader = torch.utils.data.DataLoader(inputs_dataset, batch_size=batch_size)
        return self.get_output_targets(dataloader)
