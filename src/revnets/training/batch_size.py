import typing
from typing import Literal

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.tuner.tuning import Tuner
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..context import context
from .trainer import Trainer

if typing.TYPE_CHECKING:
    from ..data import Dataset  # noqa: autoimport
from .network import Network


class TuneModel(LightningModule):
    def __init__(self, model: Network, data: Dataset) -> None:
        super().__init__()
        self.model = model
        self.data = data
        self.batch_size = 1
        self.old_batch_size = self.data.batch_size
        self.automatic_optimization = False

    def val_dataloader(self) -> DataLoader[torch.Tensor]:
        self.data.eval_batch_size = self.batch_size
        return self.data.val_dataloader()

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        return self.model.validation_step(batch, batch_idx)

    def train_dataloader(self) -> DataLoader[torch.Tensor]:
        self.data.batch_size = self.batch_size
        return self.data.train_dataloader()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.model.training_step(batch, batch_idx)

    def configure_optimizers(self) -> Optimizer:
        return self.model.configure_optimizers()


def calculate_max_batch_size(
    network: Network | LightningModule,
    data: Dataset,
    method: Literal["fit", "validate", "test", "predict"] = "validate",
) -> int:
    tune_model = TuneModel(network, data)
    data.prepare()

    trainer = Trainer(max_epochs=1, devices=1)
    tuner = Tuner(trainer)

    print(f"Calculating max {method} batch size")

    if isinstance(network, Network):
        network.do_log = False
    old_batch_size = data.batch_size
    try:
        batch_size = context.config.reconstruction_training.batch_size
        tuner.scale_batch_size(tune_model, method=method, init_val=batch_size)
    except ValueError:
        message = "Batch size of 2 does not fit in GPU, impossible to start training"
        raise Exception(message)

    if isinstance(network, Network):
        network.do_log = True
    data.batch_size = old_batch_size

    safety_factor = 2 if method == "validate" else 1
    max_batch_size = tune_model.batch_size // safety_factor
    max_batch_size |= 1  # max sure batch size at least one
    return max_batch_size
