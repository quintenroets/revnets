from dataclasses import dataclass, fields

import torch

from revnets.utils import config


@dataclass
class Metrics:
    l1_loss: torch.Tensor
    l2_loss: torch.Tensor
    # smooth_l1_loss: torch.Tensor

    def dict(self):
        return self.__dict__

    @property
    def loss(self):
        match config.loss_criterion:
            # case "smooth_l1":
            # loss = self.smooth_l1_loss
            case "l1":
                loss = self.l1_loss
            case "l2":
                loss = self.l2_loss
            case _:
                message = f"Invalid loss criterion {config.loss_criterion}"
                raise Exception(message)

        return loss

    @classmethod
    @property
    def names(cls):  # noqa
        return [field.name for field in fields(cls)]
