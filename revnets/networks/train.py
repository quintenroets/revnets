import torch

from ..data.mnist1d import Dataset
from ..utils import Path
from ..utils.trainer import Trainer
from .models import trainable
from .models.trainable import Model


class Network:
    @classmethod
    def load_trained_weights(cls, model: Model, seed: int = None):
        cls.check_weights_path(model, seed)
        weights_path = cls.get_weights_path(model, seed)
        weights = torch.load(str(weights_path))
        model.load_state_dict(weights)

    @classmethod
    def check_weights_path(cls, model: Model, seed):
        weights_path = cls.get_weights_path(model, seed)
        if not weights_path.exists():
            cls.train(model)
            cls.save_weights(model, seed)

    @classmethod
    def save_weights(cls, model: Model, seed):
        weights_path = cls.get_weights_path(model, seed)
        state_dict = model.state_dict()
        torch.save(state_dict, str(weights_path))

    @classmethod
    def get_weights_path(cls, model: Model, seed=None):
        name = model.name
        if seed is not None:
            name += f"_seed{seed}"
        path: Path = Path.weights / name
        path.create_parent()
        return path

    @classmethod
    def train(cls, model: Model):
        data = cls.dataset()
        train_model = trainable.Model(model)
        data.calibrate(train_model)
        trainer = Trainer()
        trainer.fit(train_model, data)
        trainer.test(train_model, data)

    @classmethod
    def dataset(cls) -> Dataset:
        raise NotImplementedError
