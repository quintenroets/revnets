from abc import ABC

import torch

from ..context import context
from ..data.base import Dataset
from ..models import Path
from ..utils.trainer import Trainer
from . import base
from .models import Model, trainable


class Network(base.Network, ABC):
    def create_trained_model(self) -> Model:
        model = self.create_model()
        if not self.weights_path.exists():
            self.train(model)
            self.save_weights(model)
        self.load_weights(model)
        return model

    def create_initialized_model(self) -> Model:
        torch.manual_seed(context.config.experiment.seed)
        return self.create_model()

    @classmethod
    def create_model(cls) -> Model:
        raise NotImplementedError

    def train(self, model: torch.nn.Module) -> None:
        train_model = trainable.Model(model)
        torch.manual_seed(context.config.experiment.target_network_seed)
        data = self.create_dataset()
        self.run_training(train_model, data)

    @classmethod
    def run_training(cls, model: trainable.Model, data: Dataset) -> None:
        data.calibrate(model)
        trainer = Trainer(max_epochs=context.config.target_network_training.epochs)
        trainer.fit(model, data)
        trainer.test(model, data)

    @classmethod
    def create_dataset(cls) -> Dataset:
        raise NotImplementedError

    def calculate_output_size(self) -> int:
        dataset = self.create_dataset()
        dataset.prepare()
        sample = dataset.train_val_dataset[0][0]
        inputs = sample.unsqueeze(0)
        model = self.create_initialized_model()
        outputs = model(inputs)[0]
        size = outputs.shape[-1]
        return size

    def load_weights(self, model: torch.nn.Module) -> None:
        state = torch.load(self.weights_path_str)
        model.load_state_dict(state)

    def save_weights(self, model: torch.nn.Module) -> None:
        state_dict = model.state_dict()
        torch.save(state_dict, self.weights_path_str)

    @property
    def weights_path(self) -> Path:
        config = context.config.experiment
        path: Path = (
            Path.weights
            / "trained_targets"
            / "_".join(config.network_to_reconstruct)
            / str(config.target_network_seed)
        )
        path.create_parent()
        return path

    @property
    def weights_path_str(self) -> str:
        return str(self.weights_path)
