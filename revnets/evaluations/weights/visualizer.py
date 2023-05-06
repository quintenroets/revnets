import matplotlib.pyplot as plt
import torch

from ...utils.colors import get_colors
from . import layers_mae, standardize


class Evaluator(layers_mae.Evaluator):
    def evaluate(self):
        self.standardize_networks()
        networks = {"original": self.original, "reconstruction": self.reconstruction}
        for name, network in networks.items():
            self.visualize_network_weights(network, name)
        if self.same_architecture():
            self.visualize_network_differences()

    def visualize_network_weights(self, network, name):
        layers = standardize.standardize.get_layers(network)
        for i, layer in enumerate(layers):
            weights = self.get_layer_weights(layer)
            title = f"{name} layer {i + 1} weights".capitalize()
            self.visualize_weights(weights, title)

    @classmethod
    def visualize_weights(cls, weights, title, n_show=None):
        if n_show is not None:
            weights = weights[:n_show]

        # weights = torch.transpose(weights, 0, 1)

        n_neurons = len(weights)
        colors = get_colors(n=n_neurons)
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
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
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
