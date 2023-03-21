import math

import numpy as np

from .. import evaluations
from ..networks.base import Network
from ..reconstructions.outputs_supervision import random_inputs
from . import experiment


class Experiment(experiment.Experiment):
    @classmethod
    def get_results(cls, network: Network):
        original = network.get_trained_network()

        results = {}
        max_num_samples = 100000
        for num_samples in cls.get_samples_range(stop=max_num_samples):
            scale_ratio = cls.calculate_scale_ratio(num_samples, max_num_samples)
            dataset_kwargs = dict(
                num_samples=num_samples,
                repetition_factor=scale_ratio,
                validation_ratio=0,
            )
            reconstructor = random_inputs.Reconstructor(
                original, network, dataset_kwargs=dataset_kwargs
            )
            reconstruction = reconstructor.reconstruct()
            evaluation = evaluations.evaluate(original, reconstruction, network)
            results[num_samples] = evaluation
        return results

    @classmethod
    def calculate_scale_ratio(cls, num_samples, max_num_samples):
        # repeat number of samples if less data such
        # that effective number of samples is constant
        return max_num_samples / num_samples

    @classmethod
    def get_samples_range(cls, start=1000, stop=100000, num_values=50):
        start = math.log10(start)
        stop = math.log10(stop)
        return np.logspace(start, stop, num_values, dtype=int)
