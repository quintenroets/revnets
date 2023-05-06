from dataclasses import dataclass
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from revnets.data import random
from revnets.networks.models.mediumnet import Model

from . import weights


@dataclass
class Experiment(weights.Experiment):
    def run_reconstruction(self, reconstruction):
        super().run_reconstruction(reconstruction)
        # self.visualize_inputs()
        models = {
            "reconstruction": reconstruction,
            "initialization": self.network.get_architecture(),
            "original": self.network.trained_network,
        }

        try:
            get_ipython()
            is_notebook = True
        except NameError:
            is_notebook = False
        if not is_notebook:
            for name, model in models.items():
                self.visualize_activations(model, name=name)

    def visualize_inputs(self):
        random_inputs = self.get_inputs()
        dataset = self.network.dataset()
        dataset.prepare()
        train_inputs = dataset.train_val_dataset.tensors[0]
        for inputs in (random_inputs, train_inputs):
            self.visualize(inputs)

    def visualize_activations(self, model: Model, **kwargs):
        layers = {
            "first hidden": model.layer1,
            # "second hidden": [model.layer1, model.layer2],
            # "output": model,
        }
        for layer_name, model in layers.items():
            if isinstance(model, tuple) or isinstance(model, list):
                model = torch.nn.Sequential(torch.nn.ModuleList(model))
            self.visualize_model_outputs(
                model, layer_name=layer_name, activation=False, **kwargs
            )

    def visualize_model_outputs(self, model, activation=False, **kwargs):
        inputs = self.get_inputs()
        outputs = self.get_outputs(inputs, model)
        if activation:
            outputs = F.relu(outputs)
        self.visualize(outputs, **kwargs)

    @classmethod
    def get_outputs(cls, inputs, model):
        dataset = TensorDataset(inputs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(inputs))
        return random.Dataset.get_predictions(dataloader, model)

    @classmethod
    def visualize(cls, outputs, name=None, layer_name=None):
        for out in outputs:
            plt.plot(out, linewidth=0.2, alpha=0.6)
        zero = np.zeros_like(outputs[0])
        plt.plot(zero, linewidth=0.5, color="red")
        max_feature_values = outputs.max(dim=0)[0]
        plt.plot(max_feature_values, linewidth=0.5, color="blue", marker="o")
        title = f"{name} {layer_name} layer activations".capitalize()
        plt.title(title)
        plt.show()

    @classmethod
    @cache
    def get_inputs(cls):
        n_inputs = 1000
        n_features = 40
        shape = (n_inputs, n_features)
        return random.Dataset.generate_random_inputs(shape)
