import matplotlib.pyplot as plt
import numpy as np

from ..utils.colors import get_colors
from ..utils.path import Path


class Experiment:
    @classmethod
    def run(cls):
        colors = get_colors(10)
        names = ("iterative sampling", "random sampling")
        mergers = {"mean": cls.mean, "minimum": cls.min}
        mergers = {"minimum": mergers["minimum"]}
        mergers = {"first": cls.first}
        for name, merger in mergers.items():
            for color, sampling_name in zip(colors, names):
                results = cls.load_results(sampling_name)
                results = cls.merge_results(results)
                results = cls.parse_results(results)
                cls.show_merge(results, merger, name, sampling_name, color)

            name = name.capitalize()
            plt.legend()
            plt.title(f"{name} value over 4 experiments with different random seed")
            plt.show()

    @classmethod
    def show_merge(cls, results, merger, name, sampling_name, color):
        merged_results = [(k, merger(v)) for k, v in results.items()]
        merged_results = sorted(merged_results, key=lambda v: v[0])
        x_values = [v[0] for v in merged_results]
        y_values = [v[1] for v in merged_results]
        plt.plot(x_values, y_values, label=sampling_name, color=color)
        plt.xscale("log")
        plt.xlabel("Number of train data points")
        plt.yscale("log")
        plt.ylabel(f"Reconstructed weights MAE")

    @classmethod
    def mean(cls, values: np.array):
        return values.mean()

    @classmethod
    def first(cls, values: np.array):
        return values[0]

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
    def load_results(cls, name=None):
        results_folder = Path.results / "Data requirements" / "Mininet mininet"
        if name is not None:
            results_folder /= name
        return [
            path.yaml for path in results_folder.iterdir() if "fail" not in path.stem
        ]
