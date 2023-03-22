import matplotlib.pyplot as plt
import numpy as np

from ..utils.path import Path


class Experiment:
    @classmethod
    def run(cls):
        results = cls.load_results()
        results = cls.merge_results(results)
        results = cls.parse_results(results)
        for merger in (cls.mean, cls.min):
            cls.show_merge(results, merger)

    @classmethod
    def show_merge(cls, results, merger):
        merged_results = [(k, merger(v)) for k, v in results.items()]
        merged_results = sorted(merged_results, key=lambda v: v[0])
        x_values = [v[0] for v in merged_results]
        y_values = [v[1] for v in merged_results]
        plt.plot(x_values, y_values)
        plt.xscale("log")
        plt.yscale("log")
        plt.show()

    @classmethod
    def mean(cls, values: np.array):
        return values.mean()

    @classmethod
    def min(cls, values: np.array):
        return min(values)

    @classmethod
    def parse_results(cls, results):
        return {
            int(k): np.array([float(value) for value in v]) for k, v in results.items()
        }

    @classmethod
    def merge_results(cls, results):
        merged_result = {}
        for result in results:
            for k, v in result.items():
                v = v["weights_MAE"]
                if k not in merged_result:
                    merged_result[k] = []
                merged_result[k].append(v)
        return merged_result

    @classmethod
    def load_results(cls):
        results_folder = Path.results / "Data requirements" / "mininet"
        return [path.yaml for path in results_folder.iterdir()]
