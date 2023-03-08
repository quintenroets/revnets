from dataclasses import dataclass, fields
from functools import cached_property

import torch.nn
from cacher.caches.deep_learning import Reducer
from cacher.hashing import compute_hash
from torch import nn

from ...data import Dataset, output_supervision
from ...networks.models import trainable
from ...utils import Path
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

    @cached_property
    def trained_weights_path(self):
        data: Dataset = self.network.dataset()
        hash_value = compute_hash(
            Reducer,
            self.original.__class__.__module__,
            self.reconstruction.__class__.__module__,
            self.reconstruction.state_dict(),
            data,
        )
        path = Path.weights / "reconstructions" / self.name / hash_value
        path.create_parent()
        return path

    def reconstruct_weights(self):
        if self.always_train or not self.trained_weights_path.exists():
            self.start_training()
            self.save_weights()

        state_dict = torch.load(self.trained_weights_path)
        self.reconstruction.load_state_dict(state_dict)

    def start_training(self):
        self.model = ReconstructModel(self.reconstruction)
        data = self.get_dataset()
        data.calibrate(self.model)
        trainer = Trainer()
        trainer.fit(self.model, data)
        trainer.test(self.model, data)

    def get_train_model(self):
        return ReconstructModel(self.reconstruction)

    def save_weights(self):
        state_dict = self.reconstruction.state_dict()
        torch.save(state_dict, str(self.trained_weights_path))

    def get_dataset(self):
        data: Dataset = self.network.dataset()
        dataset_module = self.get_dataset_module()
        data: Dataset = dataset_module.Dataset(data, self.original)
        data.setup("train")
        data.calibrate(self.model)
        return data

    @classmethod
    def get_dataset_module(cls):
        return output_supervision
