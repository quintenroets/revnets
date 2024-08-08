from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.nn import Module

from revnets.context import context
from revnets.evaluations.weights import layers_mae
from revnets.utils.colors import get_colors

cpu = torch.device("cpu")


@dataclass
class Evaluator(layers_mae.Evaluator):
    def evaluate(self) -> None:
        self.standardize_networks()
        networks = {"target": self.target, "reconstruction": self.reconstruction}
        if not context.config.evaluation.only_visualize_differences:
            for name, network in networks.items():
                self.visualize_network_weights(network, name)
        if self.has_same_architecture():
            self.visualize_network_differences()

    def visualize_network_weights(self, network: Module, name: str) -> None:
        layer_weights = layers_mae.extract_weights(network)
        for i, weights in enumerate(layer_weights):
            title = f"{name} layer {i + 1} weights".capitalize()
            self.visualize_layer_weights(weights, title)

    @classmethod
    def visualize_layer_weights(
        cls,
        weights: torch.Tensor,
        title: str,
        n_show: int | None = 10,
    ) -> None:
        weights = weights[:n_show].cpu()

        n_neurons = len(weights)
        colors = get_colors(number_of_colors=n_neurons)
        ax = cls.create_figure()

        for i, (neuron, color) in enumerate(zip(weights, colors, strict=False)):
            label = f"Neuron {i + 1}"
            ax.plot(neuron, color=color, label=label)

        n_features = len(weights[0])
        interval = n_features // 4
        x_ticks = [*range(0, n_features, interval), n_features - 1]
        if n_features - 2 not in x_ticks and False:
            x_ticks.insert(-1, n_features - 2)
        x_tick_labels = [str(xtick) for xtick in x_ticks[:-1]] + ["Bias weight"]
        plt.xticks(x_ticks, x_tick_labels)

        plt.title(title)
        plt.xlabel("Weight index")
        plt.ylabel("Weight Value")
        cls.show()

    def visualize_network_differences(self) -> None:
        layers = self.iterate_compared_layers()
        for i, (original, reconstructed) in enumerate(layers):
            weights = original - reconstructed
            title = f"Layer {i + 1} weight differences"
            self.visualize_layer_weights(weights, title)

    @classmethod
    def create_figure(cls) -> Any:
        _, ax = plt.subplots(figsize=(20, 10))
        return ax

    @classmethod
    def show(cls) -> None:
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.show()
