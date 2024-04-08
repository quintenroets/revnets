from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rich.pretty import pprint

from ...models import Path
from . import activations


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
    def from_values(cls, values: list[float] | NDArray[np.float64]) -> Statistics:
        values_numpy = np.array(values) * 1000
        return cls(
            mean=values_numpy.mean(), min=values_numpy.min(), std=values_numpy.std()
        )

    def __repr__(self) -> str:
        return f"{self.mean} + {self.std}    {self.min}"


@dataclass
class Experiment(activations.Experiment):
    n_inputs: int = 1000

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
            combined_results = {
                k: self.extract_statistics(v) for k, v in combined_result.items()
            }

            pprint(network)
            pprint(combined_results)

    @classmethod
    def extract_statistics(cls, values: dict[str, list[str]]) -> Statistics:
        float_values = [float(v) for v in values["weights_MAE"]]
        return Statistics.from_values(float_values)

    @classmethod
    def combine_keys(cls, items: list[dict[str, Any]]) -> dict[str, Any]:
        return {k: [item[k] for item in items] for k in items[0].keys()}
