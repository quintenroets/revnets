from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from kneed import KneeLocator
from torch.utils.data import ConcatDataset

from ...data import output_supervision
from ...data.random import Dataset
from ...utils.trainer import Trainer
from . import random_inputs
from .base import ReconstructModel


@dataclass
class Reconstructor(random_inputs.Reconstructor):
    n_rounds: int = 10
    noise_factor: float = 1 / 100
    data: Dataset = None

    def __post_init__(self):
        super().__post_init__()
        # make total samples used equal to other experiments
        num_validation_samples = int(self.num_samples * Dataset.validation_ratio)
        self.dataset_kwargs = dict(
            num_samples=num_validation_samples, validation_ratio=1
        )
        self.num_samples //= self.n_rounds

    def start_training(self):
        self.model = ReconstructModel(self.reconstruction, self.network)
        self.check_randomize()
        self.data = self.get_dataset()

        for i in range(self.n_rounds):
            self.run_round()

    def get_dataset(self):
        data = super().get_dataset()
        data.prepare()
        shape = (self.num_samples, *data.input_shape)  # noqa
        inputs = data.generate_random_inputs(shape)
        data.train_dataset = data.construct_dataset(inputs)
        return data

    def run_round(self):
        self.train_model()
        self.add_difficult_samples()

    def train_model(self):
        # early_stop_callback = EarlyStopping(
        # monitor="train l1_loss", patience=100, mode="min"
        # )
        # callbacks = [early_stop_callback]
        path = Path("weightos")
        if path.exists() and False:
            weights = torch.load(path)
            self.model.load_state_dict(weights)
        else:
            callbacks = []
            trainer = Trainer(callbacks=callbacks)
            trainer.fit(self.model, self.data)
            weights = self.model.state_dict()
            torch.save(weights, path)

    def add_difficult_samples(self):
        difficult_inputs = self.sample_difficult_inputs()
        extra_dataset = self.data.construct_dataset(difficult_inputs)
        datasets = extra_dataset, self.data.train_dataset
        self.data.train_dataset = ConcatDataset(datasets)

    def sample_difficult_inputs(self):
        difficult_inputs = self.extract_difficult_inputs()
        recombined_inputs = self.recombine(difficult_inputs)
        noise = self.data.generate_random_inputs(recombined_inputs.shape)
        new_difficult_inputs = recombined_inputs + self.noise_factor * noise
        return new_difficult_inputs

    def recombine(self, inputs):
        new_samples_shape = self.num_samples, inputs.shape[-1]
        n_inputs = len(inputs)
        new_samples = np.random.choice(range(n_inputs), size=new_samples_shape)
        # each feature value in a new sample corresponds with a feature value
        # in the corresponding feature of one of the inputs
        return inputs[new_samples, np.arange(new_samples.shape[1])]

    def extract_difficult_inputs(self):
        split = output_supervision.Split.train
        inputs = self.data.get_all_inputs(split)
        targets = self.data.get_all_targets(split)

        outputs_dataset = output_supervision.Dataset(self.data, self.model)
        outputs = outputs_dataset.add_output_supervision(split).tensors[1]

        high_loss_indices = self.get_high_loss_indices(outputs, targets)
        high_loss_inputs = inputs[high_loss_indices]
        return high_loss_inputs

    def get_high_loss_indices(self, outputs, targets):
        sorted_losses, original_indices = self.calculate_sorted_losses(outputs, targets)
        elbow = self.get_elbow(sorted_losses)
        return original_indices[:elbow]

    @classmethod
    def calculate_sorted_losses(cls, outputs, targets):
        losses = torch.nn.functional.l1_loss(outputs, targets, reduction="none")
        losses = losses.mean(dim=1)
        sorted_losses, original_indices = torch.sort(losses, descending=True)
        return sorted_losses, original_indices

    @classmethod
    def get_elbow(cls, values):
        elbow_range = range(len(values))
        elbow_result = KneeLocator(
            elbow_range, values, curve="convex", direction="decreasing"
        )
        return elbow_result.elbow
