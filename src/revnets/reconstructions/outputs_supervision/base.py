from dataclasses import dataclass, field
from functools import cached_property
from types import ModuleType
from typing import Any, Generic, TypeVar, cast

import torch
from pytorch_lightning.callbacks import EarlyStopping

from revnets.context import context
from revnets.data import Dataset, output_supervision
from revnets.models import Path
from revnets.training import Trainer

from .. import empty
from .reconstruction_model import ReconstructNetwork

T = TypeVar("T", bound=output_supervision.Dataset)


@dataclass
class Reconstructor(empty.Reconstructor, Generic[T]):
    network: ReconstructNetwork = field(init=False)
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    visualize_weights: bool = False
    visualization_interval: int = 10

    def __post_init__(self) -> None:
        super().__post_init__()
        self.dataset_kwargs = {"num_samples": context.config.sampling_data_size}

    @cached_property
    def trained_weights_path(self) -> Path:
        path = (
            Path.weights
            / "reconstructions"
            / self.name
            / self.pipeline.name
            / self.pipeline.network_factory
            / self.reconstruction.name
        )
        path.create_parent()
        return cast(Path, path)

    def reconstruct_weights(self) -> None:
        # needs to be calculated before training
        if context.config.always_train or not self.trained_weights_path.exists():
            self.start_training()
            self.save_weights()

        self.load_weights()

    def start_training(self) -> None:
        self.network = ReconstructNetwork(
            self.reconstruction,
            self.pipeline,
            visualize=self.visualize_weights,
            visualization_interval=self.visualization_interval,
        )
        data = self.get_dataset()
        self.train_model(data)

    def train_model(self, data: output_supervision.Dataset) -> None:
        patience = context.config.early_stopping_patience
        callback = EarlyStopping("train l1 loss", patience=patience, verbose=True)
        callbacks = [callback]
        trainer = Trainer(
            callbacks=callbacks,  # type: ignore[arg-type]
            max_epochs=context.config.target_network_training.epochs,
        )
        validation_ratio = context.config.validation_ratio
        if validation_ratio is not None and validation_ratio > 0:
            trainer.fit(self.network, data)
            trainer.test(self.network, data)
        else:
            data.prepare()
            train_dataloader = data.train_dataloader()
            trainer.fit(self.network, train_dataloaders=train_dataloader)

    def load_weights(self) -> None:
        state_dict = torch.load(self.trained_weights_path)
        self.reconstruction.load_state_dict(state_dict)

    def save_weights(self) -> None:
        state_dict = self.reconstruction.state_dict()
        torch.save(state_dict, str(self.trained_weights_path))

    def get_dataset(self, **kwargs: Any) -> T:
        pipeline_data: Dataset = self.pipeline.create_dataset()
        dataset_module = self.get_dataset_module()
        dataset_kwargs = self.dataset_kwargs | kwargs
        original = self.pipeline.create_initialized_network()
        data: T = dataset_module.Dataset(pipeline_data, original, **dataset_kwargs)
        data.calibrate(self.network)
        return data

    @classmethod
    def get_dataset_module(cls) -> ModuleType:
        return output_supervision
