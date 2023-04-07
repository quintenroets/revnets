from dataclasses import dataclass

import pytorch_lightning as pl
import torchmetrics
from cacher.caches.speedup_deep_learning import cache

from ...utils.trainer import Trainer
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
    def __init__(self, model1, model2):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.mae_metric = torchmetrics.MeanAbsoluteError()
        self.mse_metric = torchmetrics.MeanSquaredError()
        self.metrics = None

    def test_step(self, batch, batch_idx):
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
    def evaluate(self):
        return self.compare_outputs()

    @cache
    def compare_outputs(self):
        model = CompareModel(self.original, self.reconstruction)

        dataset = self.get_dataset()
        dataset.prepare()
        dataset.calibrate(model)

        dataloader = self.get_dataloader(dataset)

        Trainer().test(model, dataloaders=dataloader)  # noqa
        return model.metrics

    @classmethod
    def get_dataloader(cls, dataset):
        return dataset.val_dataloader()

    @classmethod
    def format_evaluation(cls, value: Metrics, **kwargs):
        values = (value.mae, value.mse)
        formatted_values = (
            super(Evaluator, cls).format_evaluation(value, **kwargs) for value in values
        )
        return FormattedMetrics(*formatted_values)
