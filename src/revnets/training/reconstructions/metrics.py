from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from simple_classproperty import classproperty
from torch import nn

from revnets.context import context

from .. import metrics


@dataclass
class Metrics(metrics.Metrics):
    shape: torch.Size
    l1_loss_sum: torch.Tensor
    l2_loss_sum: torch.Tensor
    # smooth_l1_loss: torch.Tensor
    loss: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.size = math.prod(self.shape)
        self.batch_size = self.shape[0]
        self.loss = self.loss_sum / self.size

    @property
    def l1_loss(self) -> torch.Tensor:
        return self.l1_loss_sum / self.size

    @property
    def l2_loss(self) -> torch.Tensor:
        return self.l2_loss_sum / self.size

    @property
    def loss_sum(self) -> torch.Tensor:
        match context.config.loss_criterion:
            # case "smooth_l1":
            # loss = self.smooth_l1_loss
            case "l1":
                loss = self.l1_loss_sum
            case "l2":
                loss = self.l2_loss_sum
            case _:
                message = f"Invalid loss criterion {context.config.loss_criterion}"
                raise Exception(message)
        if self.batch_size != context.config.reconstruction_training.batch_size:
            # optimizer expects a loss summed over
            # reconstruction_training.batch_size elements
            # scale loss appropriately if less effective elements in batch
            loss *= context.config.reconstruction_training.batch_size / self.batch_size
        return loss

    @classmethod
    @classproperty
    def names(cls) -> tuple[str, ...]:
        return ("l1_loss",)

    @classmethod
    def from_results(cls, outputs: torch.Tensor, targets: torch.Tensor) -> Metrics:
        return cls(
            shape=outputs.shape,
            l1_loss_sum=nn.functional.l1_loss(outputs, targets, reduction="sum"),
            l2_loss_sum=nn.functional.mse_loss(outputs, targets, reduction="sum"),
            # smooth_l1_loss=nn.functional.smooth_l1_loss(outputs, targets)
        )
