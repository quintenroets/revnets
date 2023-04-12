import matplotlib.pyplot as plt
import numpy as np

from ..utils.colors import get_colors
from ..utils.path import Path


class Experiment:
    @classmethod
    def run(cls):
        mergers = {"mean": cls.mean, "minimum": cls.min, "success rate": cls.success}
        names = ("iterative sampling", "random sampling")
        num_results = len(cls.load_results(names[0]))

        for name, merger in mergers.items():
            cls.show_plot(name, merger, num_results, names)

    @classmethod
    def show_plot(cls, name, merger, num_results, names):
        colors = get_colors(10)
        for color, sampling_name in zip(colors, names):
            results = cls.get_results(sampling_name)
            cls.show_results(results, merger, sampling_name, color)

        name = name.capitalize()
        plt.legend()
        title = (
            f"{name} value over {num_results} experiments with different random seed"
        )
        plt.title(title)
        plt.show()

    @classmethod
    def show_results(cls, results, merger, sampling_name, color):
        merged_results = [(k, merger(v)) for k, v in results.items()]
        merged_results = sorted(merged_results, key=lambda v: v[0])
        x_values = [v[0] for v in merged_results]
        y_values = [v[1] for v in merged_results]
        plt.plot(x_values, y_values, label=sampling_name, color=color)

        is_success_plot = merger == cls.success

        plt.xscale("log")
        plt.xlabel("Number of train data points")
        if not is_success_plot:
            plt.yscale("log")
        y_label = (
            "Fraction of successful reconstructions (weights MAE < 0.01)"
            if is_success_plot
            else "Reconstructed weights MAE"
        )
        plt.ylabel(y_label)

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
    def success(cls, values: np.ndarray, threshold=0.01):
        num_success = np.sum(values < threshold)
        num_total = len(values)
        return num_success / num_total

    @classmethod
    def get_results(cls, name):
        results = cls.load_results(name)
        results = cls.merge_results(results)
        results = cls.parse_results(results)
        return results

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
            path.yaml
            for path in results_folder.iterdir()
            if "fail" not in path.stem and path.is_file()
        ]
