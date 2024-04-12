from dataclasses import dataclass
from functools import cached_property
from typing import cast

import torch
from pytorch_lightning.callbacks import EarlyStopping

from revnets.context import context
from revnets.models import Path
from revnets.training import Trainer
from revnets.training.reconstructions import Network

from .. import base
from .data import DataModule


@dataclass
class Reconstructor(base.Reconstructor):
    num_samples: int = context.config.sampling_data_size

    @cached_property
    def trained_weights_path(self) -> Path:
        path = Path.weights / "reconstructions" / self.name / self.pipeline.name
        path.create_parent()
        return cast(Path, path)

    def reconstruct_weights(self) -> None:
        if context.config.always_train or not self.trained_weights_path.exists():
            self.start_training()
            self.save_weights()
        self.load_weights()

    def create_trainer(self) -> Trainer:
        patience = context.config.early_stopping_patience
        callback = EarlyStopping("train l1 loss", patience=patience, verbose=True)
        callbacks = [callback]
        max_epochs = context.config.reconstruction_training.epochs
        return Trainer(
            callbacks=callbacks,  # type: ignore[arg-type]
            max_epochs=max_epochs,
        )

    def create_train_network(self) -> Network:
        return Network(self.reconstruction)

    def start_training(self) -> None:
        data = self.create_training_data()
        network = self.create_train_network()
        trainer = self.create_trainer()
        trainer.fit(network, data)

    def load_weights(self) -> None:
        state_dict = torch.load(self.trained_weights_path)
        self.reconstruction.load_state_dict(state_dict)

    def save_weights(self) -> None:
        state_dict = self.reconstruction.state_dict()
        torch.save(state_dict, str(self.trained_weights_path))

    def create_training_data(self) -> DataModule:
        data = DataModule(pipeline=self.pipeline)
        self.add_queries(data)
        return data

    def add_queries(self, data: DataModule) -> None:
        queries = self.create_queries(self.num_samples)
        data.train.add(queries)
        validation_num_samples = int(self.num_samples * context.config.validation_ratio)
        for dataset in data.validation, data.test:
            validation_queries = self.create_queries(validation_num_samples)
            dataset.add(validation_queries)

    def create_queries(self, num_samples: int) -> torch.Tensor:
        raise NotImplementedError
