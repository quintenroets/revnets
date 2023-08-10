from typing import Any

import torch
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import optim

from revnets.networks import Network
from revnets.utils import config

from ....networks.models import trainable
from .metrics import Metrics


class ReconstructModel(trainable.Model):
    def __init__(self, model: torch.nn.Module, network: Network):
        super().__init__(model)
        self.network: Network = network
        self._learning_rate = config.lr

    def calculate_metrics(self, outputs, targets):
        return Metrics.from_results(outputs, targets)

    def training_step(self, batch, batch_idx):
        metrics: Metrics = self.obtain_metrics(batch, trainable.Phase.TRAIN)
        return self.extract_loss(metrics)

    @classmethod
    def extract_loss(cls, metrics: Metrics):
        return metrics.loss

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(), lr=self._learning_rate)

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        if value != self._learning_rate:
            self.update_learning_rate(value)
            self._learning_rate = value

    def update_learning_rate(self, learning_rate):
        try:
            optimizer = self.optimizers()
            # if training is no longer running, optimizer is an empty list
            if optimizer:
                self.update_optimizer(optimizer, learning_rate)
        except RuntimeError:
            # optimizers not yet configured
            pass

    @classmethod
    def update_optimizer(cls, optimizer: LightningOptimizer, learning_rate):
        for param_group in optimizer.param_groups:  # noqa
            param_group["lr"] = learning_rate
