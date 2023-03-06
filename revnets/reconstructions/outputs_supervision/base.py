from dataclasses import dataclass, fields
from functools import cached_property

import torch.nn
from cacher.caches.deep_learning import Reducer
from cacher.hashing import compute_hash
from torch import nn

from ...data.mnist1d import Dataset
from ...networks.models import trainable
from ...networks.models.metrics import Phase
from ...utils import Path
from ...utils.trainer import Trainer
from .. import empty


@dataclass
class Metrics:
    loss: torch.Tensor

    def dict(self):
        return self.__dict__

    @classmethod
    @property
    def names(cls):
        return [field.name for field in fields(cls)]


class ReconstructModel(trainable.Model):
    def __init__(self, original, reconstructed):
        super().__init__(reconstructed)
        self.original = original
        self.reconstructed = reconstructed
        self.do_log: bool = True

    def training_step(self, batch, batch_idx):
        metrics = self.obtain_metrics(batch, Phase.TRAIN)
        return metrics.loss

    def validation_step(self, batch, batch_idx):
        self.obtain_metrics(batch, Phase.VAL)

    def test_step(self, batch, batch_idx):
        self.obtain_metrics(batch, Phase.TEST)

    def obtain_metrics(self, batch, phase: Phase) -> Metrics:
        inputs, labels = batch
        targets = self.original(inputs)
        outputs = self.reconstructed(inputs)
        metrics = self.calculate_metrics(outputs, targets)
        self.log_metrics(metrics, phase)
        return metrics

    def calculate_metrics(self, outputs, targets):
        return Metrics(
            loss=nn.functional.mse_loss(outputs, targets),
        )


@dataclass
class Reconstructor(empty.Reconstructor):
    always_train: bool = False

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
        data: Dataset = self.network.dataset()
        model = ReconstructModel(self.original, self.reconstruction)
        data.calibrate(model)
        trainer = Trainer()
        trainer.fit(model, data)
        trainer.test(model, data)

    def save_weights(self):
        state_dict = self.reconstruction.state_dict()
        torch.save(state_dict, str(self.trained_weights_path))
