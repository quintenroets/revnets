from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch

from revnets.networks.models.mediumnet import Model

from . import experiment
from revnets.data import mnist1d

@dataclass
class Experiment(experiment.Experiment):
    def run_network(self):
        inputs = self.get_inputs()
        self.visualize(inputs)
        dataset = mnist1d.Dataset()
        dataset.prepare()
        print(len(dataset.train_val_dataset.tensors[0]))
        exit()
        self.visualize(dataset.train_val_dataset.tensors[0])
        exit()
        model = self.network.trained_network
        self.visualize_activations(model)
        self.visualize_activations2(model)
        self.visualize_outputs(model)

    def visualize_activations(self, model: Model):
        inputs = self.get_inputs()
        outputs = self.get_outputs(inputs, model.layer1)
        self.visualize(outputs, "First hidden layer activations")

    def visualize_activations2(self, model: Model):
        inputs = self.get_inputs()
        with torch.no_grad():
            outputs = model.layer1(inputs)
            outputs = model.layer2(outputs)
        self.visualize(outputs, "Second hidden layer activations")

    @classmethod
    def get_outputs(cls, inputs, model):
        with torch.no_grad():
            return model(inputs)

    def visualize_outputs(self, model):
        inputs = self.get_inputs()
        outputs = self.get_outputs(inputs, model)
        self.visualize(outputs, "Output activations")

    @classmethod
    def visualize(cls, outputs, title=None):
        for out in outputs:
            plt.plot(out, linewidth=0.2, alpha=0.6)
        if title is not None:
            plt.title(title)
        plt.show()

    @classmethod
    def get_inputs(cls):
        n_inputs = 1000
        return torch.rand(n_inputs, 40) * 0.6
        return torch.randn(n_inputs, 40)
