from dataclasses import dataclass

import matplotlib.pyplot as plt

from .. import networks
from ..evaluations.weights import standardize
from ..reconstructions.outputs_supervision import iterative_sampling_analysis
from . import experiment


@dataclass
class Experiment(experiment.Experiment):
    @classmethod
    def get_networks(cls):
        return (networks.mininet.mininet,)

    def run_network(self):
        network = self.network.trained_network
        self.visualize_network_weights(network)
        standardize.standardize.standardize(network)
        self.visualize_network_weights(network)

    def visualize_network_weights(self, network):
        layers = standardize.standardize.get_layers(network)
        for i, layer in enumerate(layers):
            weights = standardize.order.get_layer_weights(layer)
            # weights = weights.transpose(1, 0)
            self.visualize_weights(weights, i + 1)
            break

    @classmethod
    def visualize_weights(cls, weights, layer_number):
        n_neurons = len(weights)
        colors = iterative_sampling_analysis.Reconstructor.get_colors(n=n_neurons)
        fig, ax = plt.subplots(figsize=(10, 10))
        for neuron, color in zip(weights, colors):
            ax.plot(neuron, color=color)

        plt.title(f"Layer {layer_number} weights")
        plt.xlabel("Weight index")
        plt.ylabel("Weight Value")
        plt.show()
