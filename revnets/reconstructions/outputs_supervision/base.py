from dataclasses import dataclass, field, fields
from functools import cached_property

import torch.nn
from cacher.caches.deep_learning import Reducer
from cacher.hashing import compute_hash
from torch import nn

from ...data import Dataset, output_supervision
from ...networks.models import trainable
from ...utils import Path, config
from ...utils.trainer import Trainer
from .. import empty


@dataclass
class Metrics:
    l1_loss: torch.Tensor
    l2_loss: torch.Tensor

    def dict(self):
        return self.__dict__

    @property
    def loss(self):
        return self.l1_loss

    @classmethod
    @property
    def names(cls):  # noqa
        return [field.name for field in fields(cls)]


class ReconstructModel(trainable.Model):
    def calculate_metrics(self, outputs, targets):
        return Metrics(
            l1_loss=nn.functional.l1_loss(outputs, targets),
            l2_loss=nn.functional.mse_loss(outputs, targets),
        )


@dataclass
class Reconstructor(empty.Reconstructor):
    always_train: bool = False
    model: ReconstructModel = None
    dataset_kwargs: dict = field(default_factory=dict)

    @cached_property
    def trained_weights_path(self):
        data: Dataset = self.network.dataset()
        hash_value = compute_hash(
            Reducer,
            self.original.name,
            self.reconstruction.name,
            self.reconstruction.state_dict(),
            data,
        )
        path = Path.weights / "reconstructions" / self.name / hash_value
        path.create_parent()
        return path

    def reconstruct_weights(self):
        always_train = (
            self.always_train if config.always_train is None else config.always_train
        )
        if always_train or not self.trained_weights_path.exists():
            self.start_training()
            self.save_weights()

        self.load_weights()

    def start_training(self):
        self.model = ReconstructModel(self.reconstruction)
        data = self.get_dataset()
        trainer = Trainer()
        if data.validation_ratio > 0:
            trainer.fit(self.model, data)
            trainer.test(self.model, data)
        else:
            data.prepare()
            train_dataloader = data.train_dataloader()
            trainer.fit(self.model, train_dataloaders=train_dataloader)

    def get_train_model(self):
        return ReconstructModel(self.reconstruction)

    def load_weights(self):
        state_dict = torch.load(self.trained_weights_path)
        self.reconstruction.load_state_dict(state_dict)

    def save_weights(self):
        state_dict = self.reconstruction.state_dict()
        torch.save(state_dict, str(self.trained_weights_path))

    def get_dataset(self):
        data: Dataset = self.network.dataset()
        dataset_module = self.get_dataset_module()
        data: Dataset = dataset_module.Dataset(
            data, self.original, **self.dataset_kwargs  # noqa
        )
        data.calibrate(self.model)
        return data

    @classmethod
    def get_dataset_module(cls):
        return output_supervision
