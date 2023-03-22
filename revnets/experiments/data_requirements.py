import math
from dataclasses import dataclass

import cli
import numpy as np
from retry import retry

from .. import evaluations, networks
from ..reconstructions.outputs_supervision import random_inputs
from . import experiment


@dataclass
class Experiment(experiment.Experiment):
    min_num_samples: int = 1000
    max_num_samples: int = 100000
    num_values: int = 20
    execution_failed: bool = False

    @classmethod
    def get_networks(cls):
        return (networks.mininet.mininet,)

    def get_network_results(self):
        results = {}
        for num_samples in self.get_samples_range():
            try:
                results[num_samples] = self.get_sample_results(num_samples)
            except RuntimeError:
                self.execution_failed = True
                break
        return results

    @retry(RuntimeError, tries=15, delay=10, backoff=1.5)
    def get_sample_results(self, num_samples):
        cli.console.rule(str(num_samples))
        scale_ratio = self.calculate_scale_ratio(num_samples)
        dataset_kwargs = dict(
            num_samples=num_samples,
            repetition_factor=scale_ratio,
            validation_ratio=0,
        )
        reconstructor = random_inputs.Reconstructor(
            self.network, dataset_kwargs=dataset_kwargs
        )
        reconstruction = reconstructor.reconstruct()
        return evaluations.evaluate(reconstruction, self.network)

    def calculate_scale_ratio(self, num_samples):
        # repeat number of samples if less data such
        # that effective number of samples is constant
        return self.max_num_samples / num_samples

    def get_samples_range(self):
        start = math.log10(self.min_num_samples)
        stop = math.log10(self.max_num_samples)
        return np.logspace(start, stop, self.num_values, dtype=int)

    @property
    def results_path(self):
        path = super().results_path.parent / "results.yaml"
        if self.execution_failed:
            path = path.with_stem(path.stem + "_fail")
        path = path.with_nonexistent_name()
        return path
