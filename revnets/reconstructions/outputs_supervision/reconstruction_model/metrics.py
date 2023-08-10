import math
from dataclasses import dataclass

import torch
from torch import nn

from revnets.networks.models.metrics import LogMetrics
from revnets.utils import config


@dataclass
class Metrics(LogMetrics):
    shape: torch.Size
    l1_loss_sum: torch.Tensor
    l2_loss_sum: torch.Tensor
    # smooth_l1_loss: torch.Tensor

    def __post_init__(self):
        self.size = math.prod(self.shape)
        self.batch_size = self.shape[0]

    @property
    def l1_loss(self):
        return self.l1_loss_sum / self.size

    @property
    def l2_loss(self):
        return self.l2_loss_sum / self.size

    @property
    def loss_sum(self):
        match config.loss_criterion:
            # case "smooth_l1":
            # loss = self.smooth_l1_loss
            case "l1":
                loss = self.l1_loss_sum
            case "l2":
                loss = self.l2_loss_sum
            case _:
                message = f"Invalid loss criterion {config.loss_criterion}"
                raise Exception(message)
        if self.batch_size != config.batch_size:
            # optimizer expects a loss summed over config.batch_size elements
            # scale loss appropriately if less effective elements in batch
            loss *= config.batch_size / self.batch_size
        return loss

    @property
    def loss(self):
        return self.loss_sum / self.size

    @classmethod
    @property
    def names(cls):  # noqa
        return ("l1_loss",)

    @classmethod
    def from_results(cls, outputs: torch.Tensor, targets: torch.Tensor):
        return cls(
            shape=outputs.shape,
            l1_loss_sum=nn.functional.l1_loss(outputs, targets, reduction="sum"),
            l2_loss_sum=nn.functional.mse_loss(outputs, targets, reduction="sum"),
            # smooth_l1_loss=nn.functional.smooth_l1_loss(outputs, targets)
        )
