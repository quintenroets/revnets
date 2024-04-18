from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module

from revnets.utils.data import compute_targets

from .. import base


@dataclass
class Evaluator(base.Evaluator):
    n_inputs: int = 1000
    activation: bool = False

    def evaluate(self) -> None:
        self.visualize_random_inputs()
        self.visualize_train_inputs()
        models = {
            "reconstruction": self.reconstruction,
            "target": self.pipeline.target,
        }
        for name, model in models.items():
            self.visualize_network(model, name)

    def visualize_random_inputs(self) -> None:
        inputs = self.create_queries()
        ActivationsVisualizer(inputs, "random inputs").run()

    def visualize_train_inputs(self) -> None:
        inputs = self.pipeline.load_all_train_inputs()
        ActivationsVisualizer(inputs, "train inputs").run()

    def visualize_network(self, network: nn.Module, name: str) -> None:
        first_hidden = next(network.children())
        layers = {"first hidden": first_hidden}
        for layer_name, model in layers.items():
            name = f"{name} {layer_name}"
            self.visualize_model_outputs(model, name)

    def visualize_model_outputs(self, model: Module, name: str) -> None:
        inputs = self.create_queries()
        outputs = compute_targets(inputs, model)
        if self.activation:
            outputs = F.relu(outputs)  # pragma: nocover
        ActivationsVisualizer(outputs, name).run()

    def create_queries(self) -> torch.Tensor:
        # Circular import: reconstructions should import evaluations
        from revnets.reconstructions.queries.random import Reconstructor

        return Reconstructor(self.pipeline).create_queries(self.n_inputs)


@dataclass
class ActivationsVisualizer:
    values: torch.Tensor
    name: str = ""
    max_show: int = 100

    def __post_init__(self) -> None:
        self.values = self.values.reshape(self.values.shape[0], -1)
        self.values = self.values[: self.max_show]

    def run(self) -> None:
        for value in self.values:
            plt.plot(value, linewidth=0.2, alpha=0.6)
        self.visualize_zero()
        self.visualize_maximum_values()
        title = f"{self.name} layer activations".capitalize().strip()
        plt.title(title)
        plt.show()

    def visualize_maximum_values(self) -> None:
        values = self.values.max(dim=0)[0]
        plt.plot(values, linewidth=0.5, color="blue", marker="o")

    def visualize_zero(self) -> None:
        zero = np.zeros_like(self.values[0])
        plt.plot(zero, linewidth=0.5, color="red")
