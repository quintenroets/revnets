from dataclasses import dataclass, field
from typing import cast

import cli
import torch
from rich.text import Text
from torch.utils.data import ConcatDataset

from revnets.context import context
from revnets.data.correlated_features import Dataset

from .. import correlated_features
from ..base import ReconstructNetwork


@dataclass
class Reconstructor(correlated_features.Reconstructor):
    data: Dataset = field(init=False)
    n_rounds: int = context.config.n_rounds
    visualize: bool = False
    round: int = 0

    @property
    def num_samples(self) -> int:
        num_samples = self.dataset_kwargs.get(
            "num_samples", context.config.sampling_data_size
        )
        return cast(int, num_samples)

    @num_samples.setter
    def num_samples(self, value: int) -> None:
        self.dataset_kwargs["num_samples"] = value

    def __post_init__(self) -> None:
        super().__post_init__()
        # make total samples used equal to other experiments
        validation_ratio = self.data.validation_ratio
        self.num_validation_samples = int(
            context.config.sampling_data_size * validation_ratio
        )
        dataset_kwargs = dict(
            num_samples=self.num_validation_samples, validation_ratio=1
        )
        for k, v in dataset_kwargs.items():
            self.dataset_kwargs.setdefault(k, v)

        self.num_samples //= self.n_rounds

    def start_training(self) -> None:
        self.network = ReconstructNetwork(self.reconstruction, self.pipeline)
        self.data = self.get_dataset()

        total_samples = len(self.data.train_dataset) * self.n_rounds  # type: ignore[arg-type]
        for i in range(self.n_rounds):
            self.round = i
            n_samples = len(self.data.train_dataset)  # type: ignore[arg-type]
            title = f"Round {i+1}/{self.n_rounds}: {n_samples}/{total_samples} samples"
            text = Text(title, style="black")
            cli.console.rule(text)
            self.run_round()

    def get_dataset(self) -> Dataset:  # type: ignore[override]
        dataset: Dataset = super().get_dataset(num_samples=self.num_validation_samples)
        dataset.prepare()
        inputs = self.prepare_train_data_inputs(dataset)
        dataset.train_dataset = dataset.construct_dataset(inputs)
        return dataset

    def prepare_train_data_inputs(self, dataset: Dataset) -> torch.Tensor:
        shape = (self.num_samples, *dataset.input_shape)
        return dataset.generate_random_inputs(shape)

    def run_round(self) -> None:
        self.train_model(self.data)
        if not self.is_last_round:
            self.add_difficult_samples()
        if self.visualize:
            self.network.visualizer.evaluate()

    @property
    def is_last_round(self) -> bool:
        return self.round == self.n_rounds - 1

    def add_difficult_samples(self) -> None:
        assert self.data is not None
        difficult_inputs = self.get_difficult_inputs()
        extra_dataset = self.data.construct_dataset(difficult_inputs)
        assert self.data.train_dataset is not None
        datasets = extra_dataset, self.data.train_dataset
        self.data.train_dataset = ConcatDataset(datasets)

    def get_difficult_inputs(self) -> torch.Tensor:
        raise NotImplementedError
