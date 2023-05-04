from __future__ import annotations

import typing

import pytorch_lightning as pl
import torch.nn
from cacher import decorator
from cacher.caches import base
from pytorch_lightning.tuner.tuning import Tuner

from .config import config
from .trainer import Trainer

if typing.TYPE_CHECKING:
    from ..data import Dataset  # noqa: autoimport


class TuneModel(pl.LightningModule):
    def __init__(self, model, data):
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


class Reducer(base.Reducer):
    @classmethod
    def reduce_model(cls, model: torch.nn.Module):
        """We do not want a new cache entry for every weight assignment Cache
        calculated result for each model dataset and task names.
        """
        values = set()
        state_dict = model.state_dict()

        for key in state_dict.keys():
            try:
                value = state_dict[key]
                values.add(value)
            except ReferenceError:
                pass

        dimensions = {v.shape for v in values}
        return dimensions, config.devices

    @classmethod
    def reduce_datamodule(cls, data_module: pl.LightningDataModule):
        return data_module.__module__


cache = decorator.cache(Reducer)


@cache
def get_max_batch_size(model: pl.LightningModule, data: Dataset, method="validate"):
    tune_model = TuneModel(model, data)
    data.prepare()

    trainer = Trainer(max_epochs=1, devices=1)
    tuner = Tuner(trainer)

    print(f"Calculating max {method} batch size")

    model.do_log = False
    old_batch_size = data.batch_size
    try:
        tuner.scale_batch_size(
            tune_model, method=method, init_val=config.batch_size  # noqa
        )
    except ValueError:
        message = "Batch size of 2 does not fit in GPU, impossible to start training"
        raise Exception(message)

    model.do_log = True
    data.batch_size = old_batch_size

    safety_factor = (4 if config.num_devices > 1 else 2) if method == "validate" else 1
    max_batch_size = tune_model.batch_size // safety_factor
    max_batch_size |= 1  # max sure batch size at least one
    return max_batch_size
