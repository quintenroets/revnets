from dataclasses import dataclass
from typing import Any, cast

import pytorch_lightning as pl
import torch
import torchmetrics
from torch.nn import Module
from torch.utils.data import DataLoader

from revnets.data import DataModule
from revnets.training import Trainer

from .. import base


@dataclass
class Metrics:
    mse: float
    mae: float


@dataclass
class FormattedMetrics:
    mae: str
    mse: str


class CompareModel(pl.LightningModule):
    def __init__(self, model1: Module, model2: Module) -> None:
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.mae_metric = torchmetrics.MeanAbsoluteError()
        self.mse_metric = torchmetrics.MeanSquaredError()
        self.metrics: Metrics | None = None

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        inputs, labels = batch
        models = (self.model1, self.model2)
        outputs = [model(inputs) for model in models]
        for metric in self.mae_metric, self.mse_metric:
            metric.update(*outputs)

    def on_test_epoch_end(self) -> None:
        metrics = self.mse_metric, self.mae_metric
        metric_values = (metric.compute().item() for metric in metrics)
        self.metrics = Metrics(*metric_values)


class Evaluator(base.Evaluator):
    def evaluate(self) -> Metrics:
        return self.compare_outputs()

    def compare_outputs(self) -> Metrics:
        model = CompareModel(self.original, self.reconstruction)
        data = self.pipeline.load_prepared_data()
        dataloader = self.extract_dataloader(data)
        Trainer().test(model, dataloaders=dataloader)  # noqa
        return cast(Metrics, model.metrics)

    @classmethod
    def extract_dataloader(cls, data: DataModule) -> DataLoader[Any]:
        return data.val_dataloader()

    @classmethod
    def format_evaluation(cls, value: Metrics, **kwargs: Any) -> FormattedMetrics:  # type: ignore[override]
        values = (value.mae, value.mse)
        formatted_values = (
            super(Evaluator, cls).format_evaluation(value, **kwargs) for value in values
        )
        return FormattedMetrics(*formatted_values)
