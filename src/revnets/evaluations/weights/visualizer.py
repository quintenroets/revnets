from collections.abc import Iterator
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.nn import Module

from ...utils.colors import get_colors
from . import layers_mae
from .standardize import extract_layer_weights, generate_layers

cpu = torch.device("cpu")


class Evaluator(layers_mae.Evaluator):
    def evaluate(self) -> None:
        self.standardize_networks()
        networks = {"original": self.original, "reconstruction": self.reconstruction}
        for name, network in networks.items():
            self.visualize_network_weights(network, name)
        if self.has_same_architecture():
            self.visualize_network_differences()

    def visualize_network_weights(self, network: Module, name: str) -> None:
        layers = generate_layers(network)
        for i, layer in enumerate(layers):
            weights = self.extract_layer_weights(layer)
            title = f"{name} layer {i + 1} weights".capitalize()
            self.visualize_weights(weights, title)

    @classmethod
    def visualize_weights(
        cls, weights: torch.Tensor, title: str, n_show: int | None = None
    ) -> None:
        weights = weights[:n_show]

        # weights = torch.transpose(weights, 0, 1)

        n_neurons = len(weights)
        colors = get_colors(number_of_colors=n_neurons)
        ax = cls.create_figure()

        for i, (neuron, color) in enumerate(zip(weights, colors)):
            label = f"Neuron {i+1}"
            ax.plot(neuron, color=color, label=label)

        n_features = len(weights[0])
        interval = n_features // 4
        x_ticks = list(range(0, n_features, interval)) + [n_features - 1]
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
            title = f"Layer {i+1} weight differences"
            self.visualize_weights(weights, title)

    @classmethod
    def create_figure(cls) -> Any:
        _, ax = plt.subplots(figsize=(20, 10))
        return ax

    @classmethod
    def show(cls) -> None:
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.show()

    def iterate_compared_layers(
        self, device: torch.device | None = cpu
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        return super().iterate_compared_layers(device=device)

    @classmethod
    def extract_layer_weights(cls, layer: Module) -> torch.Tensor:
        return extract_layer_weights(layer, device=cpu)
