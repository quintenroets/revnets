from dataclasses import dataclass, fields

import torch.nn
from torch import nn

from ..data.mnist1d import Dataset
from ..networks.models import trainable
from ..networks.models.metrics import Phase
from ..utils.trainer import Trainer


@dataclass
class Metrics:
    loss: torch.Tensor

    def dict(self):
        return self.__dict__

    @classmethod
    @property
    def names(cls):
        return [field.name for field in fields(cls)]


class ReconstructModel(trainable.Model):
    def __init__(self, original, reconstructed):
        super().__init__(reconstructed)
        self.original = original
        self.reconstructed = reconstructed
        self.do_log: bool = True

    def training_step(self, batch, batch_idx):
        metrics = self.obtain_metrics(batch, Phase.TRAIN)
        return metrics.loss

    def validation_step(self, batch, batch_idx):
        self.obtain_metrics(batch, Phase.VAL)

    def test_step(self, batch, batch_idx):
        self.obtain_metrics(batch, Phase.TEST)

    def obtain_metrics(self, batch, phase: Phase) -> Metrics:
        inputs, labels = batch
        targets = self.original(inputs)
        outputs = self.reconstructed(inputs)
        metrics = self.calculate_metrics(outputs, targets)
        self.log_metrics(metrics, phase)
        return metrics

    def calculate_metrics(self, outputs, targets):
        return Metrics(
            loss=nn.functional.mse_loss(outputs, targets),
        )


def reconstruct(original: torch.nn.Module, reconstructed: torch.nn.Module, network):
    data: Dataset = network.dataset()
    model = ReconstructModel(original, reconstructed)
    data.calibrate(model)
    trainer = Trainer()
    trainer.fit(model, data)
    trainer.test(model, data)
