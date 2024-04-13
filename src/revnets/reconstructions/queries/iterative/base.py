from dataclasses import dataclass, field

import cli
import torch
from rich.text import Text

from revnets.context import context

from .. import correlated_features
from ..data import DataModule


@dataclass
class Reconstructor(correlated_features.Reconstructor):
    n_rounds: int = field(default_factory=lambda: context.config.n_rounds)
    visualize: bool = False
    round: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.total_samples = self.num_samples * self.n_rounds

    def start_training(self) -> None:
        data = self.create_training_data()
        for round in range(self.n_rounds):
            self.show_progress(round)
            self.run_round(data)
            self.add_difficult_inputs(data)

    def run_round(self, data: DataModule) -> None:
        trainer = self.create_trainer()
        network = self.create_train_network()
        trainer.fit(network, data)

    def show_progress(self, round: int) -> None:
        round_title = f"Round {round+1}/{self.n_rounds}"
        title = f"Round {round_title}: {self.num_samples}/{self.total_samples} samples"
        text = Text(title, style="black")
        cli.console.rule(text)

    def add_difficult_inputs(self, data: DataModule) -> None:
        queries = self.create_difficult_samples()
        data.train.add(queries)

    def create_difficult_samples(self) -> torch.Tensor:
        raise NotImplementedError  # pragma: nocover
