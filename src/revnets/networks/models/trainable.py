from collections.abc import Sequence
from typing import Any

import pytorch_lightning as pl
import torch.nn
import torchmetrics
from torch import nn

from .metrics import Metrics, Phase


class Model(pl.LightningModule):
    def __init__(self, model) -> None:
        super().__init__()
        self.model: torch.nn.Module = model
        self.do_log: bool = True

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        metrics = self.obtain_metrics(batch, Phase.TRAIN)
        return metrics.loss

    def validation_step(self, batch, batch_idx) -> None:
        self.obtain_metrics(batch, Phase.VAL)

    def test_step(self, batch, batch_idx) -> None:
        self.obtain_metrics(batch, Phase.TEST)

    def obtain_metrics(self, batch, phase: Phase) -> Metrics:
        inputs, labels = batch
        outputs = self(inputs)
        metrics = self.calculate_metrics(outputs, labels)
        self.log_metrics(metrics, phase)
        return metrics

    def calculate_metrics(self, outputs, labels: Sequence[str]):
        return Metrics(
            loss=nn.functional.cross_entropy(outputs, labels),
            accuracy=self.calculate_accuracy(outputs, labels),
        )

    @classmethod
    def calculate_accuracy(cls, outputs, labels: Sequence[str]) -> float:
        _, predictions = outputs.max(1)
        accuracy = torchmetrics.functional.accuracy(
            predictions, labels, task="multiclass", num_classes=10
        )
        return accuracy.item()

    def configure_optimizers(self) -> Any:
        return self.model.configure_optimizers()

    def log_metrics(self, metrics: Metrics, phase: Phase) -> None:
        if phase != Phase.SILENT:
            self._log_metrics(metrics, phase)

    def _log_metrics(self, metrics: Metrics, phase: Phase) -> None:
        # force epochs instead of steps in x-axis
        self.log("step", float(self.current_epoch + 1))

        for metric, value in metrics.dict().items():
            self.log_metric(phase, metric, value, prog_bar=True)

    def log_metric(self, phase: Phase, name: str, *args, **kwargs):
        name = name.replace("_", " ")
        name = f"{phase.value} {name}"
        return self.log(name, *args, **kwargs)

    def log(
        self,
        *args,
        sync_dist: bool = True,
        on_epoch: bool = True,
        on_step: bool = False,
        **kwargs,
    ):
        if self.do_log:
            return super().log(
                *args, sync_dist=sync_dist, on_epoch=on_epoch, on_step=on_step, **kwargs
            )
