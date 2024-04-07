from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from revnets import data
from revnets.networks.models.mediumnet import Model

from . import weights


@dataclass
class Experiment(weights.Experiment):
    def run_reconstruction(self, reconstruction: torch.nn.Module) -> None:
        super().run_reconstruction(reconstruction)
        self.visualize_inputs()
        models = {
            "reconstruction": reconstruction,
            "initialization": self.network.get_architecture(),
            "target": self.network.trained_network,
        }

        for name, model in models.items():
            self.visualize_activations(model, name=name)

    def visualize_inputs(self) -> None:
        random_inputs = self.get_inputs()
        dataset = self.network.dataset()
        dataset.prepare()
        train_inputs = dataset.train_val_dataset.tensors[0]
        for inputs in (random_inputs, train_inputs):
            self.visualize(inputs)

    def visualize_activations(self, model: Model, **kwargs) -> None:
        layers = {
            # "first hidden": model.layer1,
            # "second hidden": [model.layer1, model.layer2],
            "output": model
        }
        for layer_name, model in layers.items():
            if isinstance(model, tuple) or isinstance(model, list):
                model = torch.nn.Sequential(torch.nn.ModuleList(model))
            self.visualize_model_outputs(
                model, layer_name=layer_name, activation=False, **kwargs
            )

    def visualize_model_outputs(
        self, model, activation: bool = False, **kwargs
    ) -> None:
        inputs = self.get_inputs()
        outputs = self.get_outputs(inputs, model)
        if activation:
            outputs = F.relu(outputs)
        self.visualize(outputs, **kwargs)

    @classmethod
    def get_outputs(cls, inputs, model):
        dataset = TensorDataset(inputs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(inputs))
        return data.random.Dataset.get_predictions(dataloader, model)

    @classmethod
    def visualize(
        cls,
        outputs,
        name: str | None = None,
        layer_name: str | None = None,
        show_zero: bool = False,
    ) -> None:
        for out in outputs:
            plt.plot(out, linewidth=0.2, alpha=0.6)
        if show_zero:
            zero = np.zeros_like(outputs[0])
            plt.plot(zero, linewidth=0.5, color="red")
            max_feature_values = outputs.max(dim=0)[0]
            plt.plot(max_feature_values, linewidth=0.5, color="blue", marker="o")
        title = f"{name} network {layer_name} layer activations".capitalize()
        plt.title(title)
        plt.show()

    def get_inputs(self):
        dataset = self.network.dataset()
        dataset.prepare()
        item = dataset.train_val_dataset[0]
        inputs, targets = item
        n_features = inputs.view(-1).shape[0]
        n_inputs = 10000
        shape = (n_inputs, n_features)
        dataset_module = data.random
        # dataset_module = data.arbitrary_correlated_features
        dataset = dataset_module.Dataset(dataset, None)
        return dataset.generate_random_inputs(shape)
