import matplotlib.pyplot as plt
import torch

from ...utils.colors import get_colors
from . import layers_mae, standardize


class Evaluator(layers_mae.Evaluator):
    def evaluate(self):
        self.standardize_networks()
        self.visualize_network_weights(self.original)
        self.visualize_network_weights(self.reconstruction)
        if self.same_architecture():
            self.visualize_network_differences()

    def visualize_network_weights(self, network):
        layers = standardize.standardize.get_layers(network)
        for i, layer in enumerate(layers):
            weights = self.get_layer_weights(layer)
            title = f"Layer {i + 1} weights"
            self.visualize_weights(weights, title)

    @classmethod
    def visualize_weights(cls, weights, title, n_show=None):
        if n_show is not None:
            weights = weights[:n_show]

        n_neurons = len(weights)
        colors = get_colors(n=n_neurons)
        ax = cls.create_figure()

        for i, (neuron, color) in enumerate(zip(weights, colors)):
            label = f"Neuron {i}"
            ax.plot(neuron, color=color, label=label)

        plt.title(title)
        plt.xlabel("Weight index")
        plt.ylabel("Weight Value")
        cls.show()

    def visualize_network_differences(self):
        layers = self.iterate_compared_layers()
        for i, (original, reconstructed) in enumerate(layers):
            weights = original - reconstructed
            title = f"Layer {i+1} weight differences"
            self.visualize_weights(weights, title)

    @classmethod
    def create_figure(cls):
        _, ax = plt.subplots(figsize=(20, 10))
        return ax

    @classmethod
    def show(cls):
        # cls.set_fullscreen()
        plt.legend()
        plt.show()

    @classmethod
    def set_fullscreen(cls):
        mng = plt.get_current_fig_manager()
        maximized_size = mng.window.maxsize()
        mng.resize(*maximized_size)

    @classmethod
    @property
    def device(cls):
        return torch.device("cpu")

    def iterate_compared_layers(self, device=None):
        return super().iterate_compared_layers(device=self.device)

    @classmethod
    def get_layer_weights(cls, layer):
        return standardize.order.get_layer_weights(layer, device=cls.device)
