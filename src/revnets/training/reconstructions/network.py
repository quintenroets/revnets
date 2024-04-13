import torch

from revnets.context import context

from .. import network
from .metrics import Metrics


class Network(network.Network[Metrics]):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = context.config.reconstruction_training.learning_rate,
    ) -> None:
        super().__init__(model, learning_rate=learning_rate)

    def calculate_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Metrics:
        return Metrics.from_results(outputs, targets)
