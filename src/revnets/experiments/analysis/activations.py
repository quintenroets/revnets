from dataclasses import dataclass
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import TensorDataset

from revnets.data import random

from . import weights


@dataclass
class Experiment(weights.Experiment):
    n_inputs: int = 10000

    def run_reconstruction(self, reconstruction: torch.nn.Module) -> None:
        super().run_reconstruction(reconstruction)
        self.visualize_inputs()
        models = {
            "reconstruction": reconstruction,
            "initialization": self.pipeline.create_initialized_network(),
            "target": self.pipeline.create_trained_network(),
        }

        for name, model in models.items():
            self.visualize_activations(model, name=name)

    def visualize_inputs(self) -> None:
        random_inputs = self.create_inputs()
        dataset = self.pipeline.create_dataset()
        dataset.prepare()
        train_val_dataset = cast(TensorDataset, dataset.train_val_dataset)
        train_inputs = train_val_dataset.tensors[0]
        for inputs in (random_inputs, train_inputs):
            self.visualize(inputs)

    def visualize_activations(self, network: Module, **kwargs: Any) -> None:
        first_hidden = next(network.children())
        layers = {"first hidden": first_hidden}
        for layer_name, model in layers.items():
            if isinstance(model, tuple) or isinstance(model, list):
                model = torch.nn.Sequential(torch.nn.ModuleList(model))
            self.visualize_model_outputs(
                model, layer_name=layer_name, activation=False, **kwargs
            )

    def visualize_model_outputs(
        self, model: Module, activation: bool = False, **kwargs: Any
    ) -> None:
        inputs = self.create_inputs()
        outputs = self.extract_outputs(inputs, model)
        if activation:
            outputs = F.relu(outputs)
        self.visualize(outputs, **kwargs)

    @classmethod
    def extract_outputs(cls, inputs: torch.Tensor, model: Module) -> torch.Tensor:
        dataset = TensorDataset(inputs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(inputs))
        return random.Dataset.get_predictions(dataloader, model)

    @classmethod
    def visualize(
        cls,
        outputs: torch.Tensor,
        name: str | None = None,
        layer_name: str | None = None,
    ) -> None:
        for out in outputs:
            plt.plot(out, linewidth=0.2, alpha=0.6)
        zero = np.zeros_like(outputs[0])
        plt.plot(zero, linewidth=0.5, color="red")
        max_feature_values = outputs.max(dim=0)[0]
        plt.plot(max_feature_values, linewidth=0.5, color="blue", marker="o")
        title = f"{name} {layer_name} layer activations".capitalize()
        plt.title(title)
        plt.show()

    def create_inputs(self) -> torch.Tensor:
        dataset = self.pipeline.create_dataset()
        dataset.prepare()
        item = dataset.train_val_dataset[0]
        inputs, targets = item
        n_features = inputs.shape[0]
        shape = (self.n_inputs, n_features)
        return random.Dataset(original_dataset=dataset).generate_random_inputs(shape)
