from __future__ import annotations

import typing

import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner

from ..context import context
from .trainer import Trainer

if typing.TYPE_CHECKING:
    from ..data import Dataset  # noqa: autoimport


class TuneModel(pl.LightningModule):
    def __init__(self, model, data) -> None:
        super().__init__()
        self.model: pl.LightningModule = model
        self.data = data
        self.batch_size = 1
        self.old_batch_size = self.data.batch_size
        self.automatic_optimization = False

    def val_dataloader(self):
        self.data.eval_batch_size = self.batch_size
        return self.data.val_dataloader()

    def validation_step(self, batch, batch_idx):
        return self.model.validation_step(batch, batch_idx)

    def train_dataloader(self):
        self.data.batch_size = self.batch_size
        return self.data.train_dataloader()

    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch, batch_idx)

    def configure_optimizers(self):
        return self.model.configure_optimizers()


def get_max_batch_size(
    model: pl.LightningModule, data: Dataset, method: str = "validate"
):
    tune_model = TuneModel(model, data)
    data.prepare()

    trainer = Trainer(max_epochs=1, devices=1)
    tuner = Tuner(trainer)

    print(f"Calculating max {method} batch size")

    model.do_log = False
    old_batch_size = data.batch_size
    try:
        batch_size = context.config.reconstruction_training.batch_size
        tuner.scale_batch_size(tune_model, method=method, init_val=batch_size)
    except ValueError:
        message = "Batch size of 2 does not fit in GPU, impossible to start training"
        raise Exception(message)

    model.do_log = True
    data.batch_size = old_batch_size

    safety_factor = 2 if method == "validate" else 1
    max_batch_size = tune_model.batch_size // safety_factor
    max_batch_size |= 1  # max sure batch size at least one
    return max_batch_size
