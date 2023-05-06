from dataclasses import dataclass

import matplotlib.pyplot as plt

from revnets import data

from .. import experiment


@dataclass
class Experiment(experiment.Experiment):
    def run(self):
        original_data = data.mnist.Dataset()
        import pdb

        pdb.set_trace()
        exit()
        train_inputs = data.random.Dataset(original_data, None).get_train_inputs()
        Dataset = data.correlated_features.Dataset(original_data, None)
        sampled_inputs = Dataset.generate_random_inputs(train_inputs.shape)
        for inputs in train_inputs, sampled_inputs:
            plt.figure(figsize=(20, 20))
            for sample in inputs:
                plt.plot(sample, linewidth=0.5, alpha=0.5)
            plt.show()
