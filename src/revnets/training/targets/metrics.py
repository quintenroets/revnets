from dataclasses import dataclass

from .. import metrics


@dataclass
class Metrics(metrics.Metrics):
    accuracy: float
