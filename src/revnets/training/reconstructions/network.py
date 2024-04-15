from typing import cast

import torch
from pytorch_lightning.core.optimizer import LightningOptimizer

from revnets.context import context

from .. import network
from .metrics import Metrics


class Network(network.Network[Metrics]):
    def __init__(self, model: torch.nn.Module) -> None:
        learning_rate = context.config.reconstruction_training.learning_rate
        super().__init__(model, learning_rate=learning_rate)

    def calculate_metrics(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Metrics:
        return Metrics.from_results(outputs, targets)

    def update_learning_rate(self, learning_rate: float) -> None:
        try:
            optimizer = self.optimizers()
            # if training is no longer running, optimizer is an empty list
            if optimizer:  # pragma: nocover
                typed_optimizer = cast(LightningOptimizer, optimizer)
                self.update_optimizer(typed_optimizer, learning_rate)
        except RuntimeError:  # pragma: nocover
            # optimizers not yet configured
            pass

    @classmethod
    def update_optimizer(
        cls, optimizer: LightningOptimizer, learning_rate: float
    ) -> None:
        for param_group in optimizer.param_groups:  # noqa
            param_group["lr"] = learning_rate

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        if value != self._learning_rate:
            self.update_learning_rate(value)
            self._learning_rate = value
