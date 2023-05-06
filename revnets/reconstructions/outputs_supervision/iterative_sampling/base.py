from dataclasses import dataclass

import cli
from rich.text import Text
from torch.utils.data import ConcatDataset

from revnets.data.random import Dataset
from revnets.utils import config

from .. import correlated_features
from ..base import ReconstructModel


@dataclass
class Reconstructor(correlated_features.Reconstructor):
    n_rounds: int = config.n_rounds or 2
    data: Dataset = None
    visualize: bool = False
    round: int = 0

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
            self.round = i
            n_samples = len(self.data.train_dataset)
            title = f"Round {i+1}/{self.n_rounds}: {n_samples}/{total_samples} samples"
            text = Text(title, style="black")
            cli.console.rule(text)
            self.run_round()

    def get_dataset(self):
        data = super().get_dataset(num_samples=self.num_validation_samples)
        data.prepare()
        inputs = self.prepare_train_data_inputs(data)
        data.train_dataset = data.construct_dataset(inputs)
        return data

    def prepare_train_data_inputs(self, data):
        shape = (self.num_samples, *data.input_shape)  # noqa
        return data.generate_random_inputs(shape)

    def run_round(self):
        self.train_model(self.data)
        self.check_randomize()
        if not self.last_round:
            self.add_difficult_samples()
        if self.visualize:
            self.model.visualizer.evaluate()

    @property
    def last_round(self):
        return self.round == self.n_rounds - 1

    def add_difficult_samples(self):
        difficult_inputs = self.get_difficult_inputs()
        extra_dataset = self.data.construct_dataset(difficult_inputs)
        datasets = extra_dataset, self.data.train_dataset
        self.data.train_dataset = ConcatDataset(datasets)

    def get_difficult_inputs(self):
        raise NotImplementedError
