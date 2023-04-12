import math
from dataclasses import dataclass

import cli
import numpy as np
import torch

from .. import evaluations, networks
from . import experiment


@dataclass
class Experiment(experiment.Experiment):
    min_num_samples: int = 1000
    max_num_samples: int = 100000
    num_values: int = 20
    execution_failed: bool = False
    partial_results: dict = None

    @classmethod
    def get_networks(cls):
        return (networks.mininet.mininet,)

    def get_network_results(self):
        self.load_partial_results()
        results = {}
        for num_samples in self.get_samples_range():
            if str(num_samples) not in self.partial_results:
                try:
                    results[num_samples] = self.get_sample_results(num_samples)
                except RuntimeError:
                    self.execution_failed = True
                    break
        return results

    def load_partial_results(self):
        partial_result_path = self.get_partial_result_path()
        if partial_result_path is not None:
            self.partial_results = partial_result_path.yaml
            partial_result_path.unlink()
        else:
            self.partial_results = {}

    def get_sample_results(self, num_samples):
        torch.cuda.empty_cache()  # minimize cuda errors
        cli.console.rule(str(num_samples))
        scale_ratio = self.calculate_scale_ratio(num_samples)
        dataset_kwargs = dict(
            num_samples=num_samples,
            repetition_factor=scale_ratio,
            validation_ratio=0,
        )
        technique = self.get_techniques()[0]
        reconstructor = technique.Reconstructor(
            self.network, dataset_kwargs=dataset_kwargs, randomize_training=True
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
    def results_folder(self):
        return super().results_path.parent

    @classmethod
    @property
    def failure_keyword(cls):
        return "_fail"

    @property
    def results_path(self):
        path = self.results_folder / "results.yaml"
        if self.execution_failed:
            path = path.with_stem(path.stem + self.failure_keyword)
        path = path.with_nonexistent_name()
        return path

    def get_partial_result_path(self):
        partial_result_paths = self.results_folder.rglob(f"*{self.failure_keyword}*")
        return next(partial_result_paths, None)

    def save(self, results):
        self.results_path.yaml = self.partial_results | self.serialize_results(results)
