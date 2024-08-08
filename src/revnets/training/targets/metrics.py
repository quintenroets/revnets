from dataclasses import dataclass

from revnets.training import metrics


@dataclass
class Metrics(metrics.Metrics):
    accuracy: float
