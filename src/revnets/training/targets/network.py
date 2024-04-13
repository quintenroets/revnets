import torch
import torchmetrics
from torch import nn

from revnets.context import context

from .. import network
from .metrics import Metrics


class Network(network.Network[Metrics]):
    def __init__(self, model: nn.Module, do_log: bool = True) -> None:
        learning_rate = context.config.target_network_training.learning_rate
        super().__init__(model, learning_rate, do_log)

    def calculate_metrics(self, outputs: torch.Tensor, labels: torch.Tensor) -> Metrics:
        return Metrics(
            loss=nn.functional.cross_entropy(outputs, labels),
            accuracy=self.calculate_accuracy(outputs, labels),
        )

    @classmethod
    def calculate_accuracy(cls, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        _, predictions = outputs.max(1)
        accuracy = torchmetrics.functional.accuracy(
            predictions, labels, task="multiclass", num_classes=10
        )
        return accuracy.item()
