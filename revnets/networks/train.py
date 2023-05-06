from abc import ABC

import torch

from ..data.base import Dataset
from ..utils import Path
from ..utils.trainer import Trainer
from . import base
from .models import trainable


class Network(base.Network, ABC):
    @classmethod
    def load_trained_weights(cls, model: torch.nn.Module, seed: int = None):
        cls.check_weights_path(model, seed)
        weights_path = cls.get_weights_path(model, seed)
        weights = torch.load(str(weights_path))
        model.load_state_dict(weights)

    @classmethod
    def check_weights_path(cls, model: torch.nn.Module, seed):
        weights_path = cls.get_weights_path(model, seed)
        if not weights_path.exists():
            cls.train(model)
            cls.save_weights(model, seed)

    @classmethod
    def save_weights(cls, model: torch.nn.Module, seed):
        weights_path = cls.get_weights_path(model, seed)
        state_dict = model.state_dict()
        torch.save(state_dict, str(weights_path))

    @classmethod
    def get_weights_path(cls, model: torch.nn.Module, seed=None):
        seed_name = f"seed_{seed or 'none'}"
        path: Path = (
            Path.weights / "trained_blackbox" / cls.name / model.name / seed_name
        )
        path.create_parent()
        return path

    @classmethod
    def train(cls, model: torch.nn.Module):
        data = cls.dataset()
        train_model = trainable.Model(model)
        data.calibrate(train_model)
        trainer = Trainer()
        trainer.fit(train_model, data)
        trainer.test(train_model, data)

    @classmethod
    def dataset(cls) -> Dataset:
        raise NotImplementedError
