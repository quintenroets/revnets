from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from revnets import data

from .. import experiment


@dataclass
class Experiment(experiment.Experiment):
    n_show: int = 300

    def run(self) -> None:
        original_data = data.mnist1d.Dataset()
        train_inputs = data.random.Dataset(original_data, None).get_train_inputs()
        Dataset = data.correlated_features.Dataset(original_data, None)
        sampled_inputs = Dataset.generate_random_inputs(train_inputs.shape)
        n_samples = len(train_inputs)
        visualize_indices = np.random.choice(
            np.arange(n_samples), self.n_show, replace=False
        )
        train_inputs = train_inputs[visualize_indices]
        sampled_inputs = sampled_inputs[visualize_indices]

        named_inputs = {"Train inputs": train_inputs, "Sampled inputs": sampled_inputs}

        for name, inputs in named_inputs.items():
            plt.figure(figsize=(20, 20))
            for sample in inputs:
                plt.plot(sample, linewidth=0.5, alpha=0.5)
            plt.title(name)
            plt.show()
