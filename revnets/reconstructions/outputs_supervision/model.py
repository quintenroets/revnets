from dataclasses import dataclass, fields
from typing import Any

import torch.nn
from torch import nn, optim

from revnets.evaluations import weights

from ...evaluations.weights.mae import Evaluator
from ...networks.models import trainable


@dataclass
class Metrics:
    l1_loss: torch.Tensor
    l2_loss: torch.Tensor

    def dict(self):
        return self.__dict__

    @property
    def loss(self):
        return self.l1_loss
        return self.l2_loss

    @classmethod
    @property
    def names(cls):  # noqa
        return [field.name for field in fields(cls)]


class ReconstructModel(trainable.Model):
    def __init__(
        self,
        model,
        network,
        visualize=False,
        visualization_interval: int = 10,
        clip_value: float = 10,
    ):
        super().__init__(model)
        self.network = network
        self.visualization_interval = visualization_interval
        self.visualizer = weights.visualizer.Evaluator(self.model, self.network)
        self.do_visualize = visualize
        self.visualize(before_training=True)
        self.clip_value = clip_value

    def calculate_metrics(self, outputs, targets):
        return Metrics(
            l1_loss=nn.functional.l1_loss(outputs, targets),
            l2_loss=nn.functional.mse_loss(outputs, targets),
        )

    def training_step(self, batch, batch_idx):
        metrics = self.obtain_metrics(batch, trainable.Phase.TRAIN)
        self.clip_weights()
        return metrics.loss

    def clip_weights(self):
        for param in self.parameters():
            param.data.clamp_(-self.clip_value, self.clip_value)

    def log_weights_MAE(self) -> None:
        evaluator = Evaluator(self.model, self.network)
        mae = evaluator.evaluate()
        if mae is not None:
            self.log_weights_MAE_value(mae)

    def log_weights_MAE_value(self, mae):
        scales = {"": 0, "milli ": 3, "micro ": 6, "nano ": 9}
        scale_name = ""
        log_scale = 0
        for scale_name, log_scale in scales.items():
            scale = 10 ** (-log_scale)
            if mae > scale:
                break

        mae = mae * 10**log_scale

        self.log(f"weights {scale_name}MAE", mae, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        self.visualize()
        if self.trainer.current_epoch % 1 == 0:
            self.log_weights_MAE()

    def visualize(self, before_training=False):
        if self.do_visualize:
            should_visualize = before_training or self.is_visualization_epoch
            if should_visualize:
                self.visualizer.evaluate()

    @property
    def is_visualization_epoch(self):
        return (self.trainer.current_epoch + 1) % self.visualization_interval == 0

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(), lr=self.model.learning_rate)
