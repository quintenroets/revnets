from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from rich.pretty import pprint
from torch.utils.data import TensorDataset

from revnets.data import random
from revnets.networks.models.mediumnet import Model

from ...models import Path
from . import weights


@dataclass
class Statistics:
    mean: float
    std: float
    min: float

    def __post_init__(self) -> None:
        if self.mean > 0.1:
            self.mean = round(self.mean, 1)
        if self.std > 0.1:
            self.std = round(self.std, 1)
        if self.min > 0.1:
            self.min = round(self.min, 1)

    @classmethod
    def from_values(cls, values: list | np.ndarray):
        values = np.array(values)
        values *= 1000
        return cls(mean=values.mean(), min=values.min(), std=values.std())

    def __repr__(self) -> str:
        return f"{self.mean} + {self.std}    {self.min}"


@dataclass
class Experiment(weights.Experiment):
    def run(self) -> None:
        experiment_path = Path.results / "Experiment"
        for network_path in experiment_path.iterdir():
            network = network_path.stem
            results = [path.yaml for path in network_path.iterdir() if path.is_file()]
            if len(results) < 10:
                print(network)
                print(len(results))
                continue
            combined_result = self.combine_keys(results)
            combined_result = {
                k: self.combine_keys(v) for k, v in combined_result.items()
            }
            combined_result = {
                k.replace("Outputs supervision ", ""): v
                for k, v in combined_result.items()
            }
            results = {k: self.extract_mae(v) for k, v in combined_result.items()}

            print(network)
            pprint(results)
        exit()

    def extract_mae(self, values):
        values = values["weights_MAE"]
        values = [float(v) for v in values]
        return Statistics.from_values(values)

    @classmethod
    def combine_keys(cls, items):
        return {k: [item[k] for item in items] for k in items[0].keys()}

    def run_reconstruction(self, reconstruction) -> None:
        super().run_reconstruction(reconstruction)
        # self.visualize_inputs()
        models = {
            "reconstruction": reconstruction,
            "initialization": self.network.get_architecture(),
            "original": self.network.trained_network,
        }

        if config.is_notebook:
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
        return random.Dataset.get_predictions(dataloader, model)

    @classmethod
    def visualize(
        cls, outputs, name: str | None = None, layer_name: str | None = None
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

    def get_inputs(self):
        dataset = self.network.dataset()
        dataset.prepare()
        item = dataset.train_val_dataset[0]
        inputs, targets = item
        n_features = inputs.shape[0]
        n_inputs = 1000
        shape = (n_inputs, n_features)
        return random.Dataset(dataset, None).generate_random_inputs(shape)
