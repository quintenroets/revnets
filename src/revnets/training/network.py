from typing import Any, Generic, TypeVar, cast

import pytorch_lightning as pl
import torch
from torch import nn, optim

from revnets.models import Phase

from .metrics import Metrics as BaseMetrics

Metrics = TypeVar("Metrics", bound=BaseMetrics)


class Network(pl.LightningModule, Generic[Metrics]):
    def __init__(
        self, model: nn.Module, learning_rate: float, do_log: bool = True
    ) -> None:
        super().__init__()
        self._learning_rate = learning_rate
        self.model = model
        self.do_log = do_log

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.model(inputs)
        return cast(torch.Tensor, outputs)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        metrics = self.obtain_metrics(batch, Phase.TRAIN)
        return metrics.loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.obtain_metrics(batch, Phase.VAL)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.obtain_metrics(batch, Phase.TEST)

    def obtain_metrics(self, batch: torch.Tensor, phase: Phase) -> Metrics:
        inputs, labels = batch
        outputs = self(inputs)
        metrics = self.calculate_metrics(outputs, labels)
        self.log_metrics(metrics, phase)
        return metrics

    def calculate_metrics(self, outputs: torch.Tensor, labels: torch.Tensor) -> Metrics:
        raise NotImplementedError

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def log_metrics(self, metrics: Metrics, phase: Phase) -> None:
        if phase != Phase.SILENT:
            self._log_metrics(metrics, phase)

    def _log_metrics(self, metrics: Metrics, phase: Phase) -> None:
        # force epochs instead of steps in x-axis
        self.log("step", float(self.current_epoch + 1))

        for metric, value in metrics.dict().items():
            self.log_metric(phase, metric, value, prog_bar=True)

    def log_metric(self, phase: Phase, name: str, value: float, **kwargs: Any) -> None:
        name = name.replace("_", " ")
        name = f"{phase.value} {name}"
        return self.log(name, value, **kwargs)

    def log(  # type: ignore[override]
        self,
        name: str,
        value: float,
        sync_dist: bool = True,
        on_epoch: bool = True,
        on_step: bool = False,
        **kwargs: Any,
    ) -> None:
        if self.do_log:
            return super().log(
                name,
                value,
                sync_dist=sync_dist,
                on_epoch=on_epoch,
                on_step=on_step,
                **kwargs,
            )
