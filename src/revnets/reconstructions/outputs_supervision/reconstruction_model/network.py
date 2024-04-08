from typing import cast

import torch
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import optim
from torch.optim import Optimizer

from revnets.context import context
from revnets.pipelines import Pipeline
from revnets.training import Network, Phase

from .metrics import Metrics


class ReconstructNetwork(Network):
    def __init__(self, model: torch.nn.Module, pipeline: Pipeline) -> None:
        super().__init__(model, learning_rate=0)
        self.pipeline = pipeline
        self._learning_rate = context.config.reconstruction_training.learning_rate

    def calculate_metrics(  # type: ignore[override]
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Metrics:
        return Metrics.from_results(outputs, targets)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        untyped_metrics = self.obtain_metrics(batch, Phase.TRAIN)
        metrics = cast(Metrics, untyped_metrics)
        return self.extract_loss(metrics)

    @classmethod
    def extract_loss(cls, metrics: Metrics) -> torch.Tensor:
        return metrics.loss

    def configure_optimizers(self) -> Optimizer:
        return optim.Adam(self.parameters(), lr=self._learning_rate)

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        if value != self._learning_rate:
            self.update_learning_rate(value)
            self._learning_rate = value

    def update_learning_rate(self, learning_rate: float) -> None:
        try:
            optimizer = self.optimizers()
            # if training is no longer running, optimizer is an empty list
            if optimizer:
                typed_optimizer = cast(LightningOptimizer, optimizer)
                self.update_optimizer(typed_optimizer, learning_rate)
        except RuntimeError:
            # optimizers not yet configured
            pass

    @classmethod
    def update_optimizer(
        cls, optimizer: LightningOptimizer, learning_rate: float
    ) -> None:
        for param_group in optimizer.param_groups:  # noqa
            param_group["lr"] = learning_rate
