from dataclasses import dataclass

import cli
from rich.text import Text

from revnets.data.random import Dataset
from revnets.utils import config

from .. import random_inputs
from ..base import ReconstructModel


@dataclass
class Reconstructor(random_inputs.Reconstructor):
    n_rounds: int = config.n_rounds or 2
    data: Dataset = None
    visualize: bool = False

    @property
    def num_samples(self):
        return self.dataset_kwargs.get("num_samples", config.sampling_data_size)

    @num_samples.setter
    def num_samples(self, value):
        self.dataset_kwargs["num_samples"] = value

    def __post_init__(self):
        super().__post_init__()
        # make total samples used equal to other experiments
        self.num_validation_samples = int(self.num_samples * Dataset.validation_ratio)
        dataset_kwargs = dict(
            num_samples=self.num_validation_samples, validation_ratio=1
        )
        for k, v in dataset_kwargs.items():
            self.dataset_kwargs.setdefault(k, v)

        self.num_samples //= self.n_rounds

    def start_training(self):
        self.model = ReconstructModel(self.reconstruction, self.network)
        self.check_randomize()
        self.data = self.get_dataset()

        total_samples = len(self.data.train_dataset) * self.n_rounds
        for i in range(self.n_rounds):
            n_samples = len(self.data.train_dataset)
            title = f"Round {i+1}/{self.n_rounds}: {n_samples}/{total_samples} samples"
            text = Text(title, style="black")
            cli.console.rule(text)
            self.run_round()

    def get_dataset(self):
        data = super().get_dataset(num_samples=self.num_validation_samples)
        data.prepare()
        shape = (self.num_samples, *data.input_shape)  # noqa
        inputs = data.generate_random_inputs(shape)
        data.train_dataset = data.construct_dataset(inputs)
        return data

    def run_round(self):
        self.train_model(self.data)
        self.check_randomize()
        self.add_difficult_samples()
        if self.visualize:
            self.model.visualizer.evaluate()

    def add_difficult_samples(self):
        raise NotImplementedError