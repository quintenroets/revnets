from dataclasses import dataclass, field

import torch
from cacher.caches.deep_learning import Reducer
from cacher.hashing import compute_hash
from pytorch_lightning.callbacks import EarlyStopping

from ...data import Dataset, output_supervision
from ...utils import Path, config
from ...utils.trainer import Trainer
from .. import empty
from .reconstruction_model import ReconstructModel


@dataclass
class Reconstructor(empty.Reconstructor):
    always_train: bool = False
    model: ReconstructModel = None
    dataset_kwargs: dict = field(default_factory=dict)
    trained_weights_path: Path = None
    visualize_weights: bool = False
    visualization_interval: int = 10

    def __post_init__(self):
        super().__post_init__()
        self.dataset_kwargs = {"num_samples": config.sampling_data_size}

    def get_trained_weights_path(self):
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
        # needs to be calculated before training
        self.trained_weights_path = self.get_trained_weights_path()
        always_train = (
            self.always_train if config.always_train is None else config.always_train
        )
        if always_train or not self.trained_weights_path.exists():
            self.start_training()
            self.save_weights()

        self.load_weights()

    def start_training(self):
        self.model = ReconstructModel(
            self.reconstruction,
            self.network,
            visualize=self.visualize_weights,
            visualization_interval=self.visualization_interval,
        )
        data = self.get_dataset()
        self.train_model(data)

    def train_model(self, data):
        patience = config.early_stopping_patience
        callback = EarlyStopping("train l1 loss", patience=patience, verbose=True)
        callbacks = [callback]
        trainer = Trainer(callbacks=callbacks, max_epochs=config.epochs)
        if data.validation_ratio > 0:
            trainer.fit(self.model, data)
            trainer.test(self.model, data)
        else:
            data.prepare()
            train_dataloader = data.train_dataloader()
            trainer.fit(self.model, train_dataloaders=train_dataloader)

    def load_weights(self):
        state_dict = torch.load(self.trained_weights_path)
        self.reconstruction.load_state_dict(state_dict)

    def save_weights(self):
        state_dict = self.reconstruction.state_dict()
        torch.save(state_dict, str(self.trained_weights_path))

    def get_dataset(self, **kwargs):
        data: Dataset = self.network.dataset()
        dataset_module = self.get_dataset_module()
        dataset_kwargs = self.dataset_kwargs | kwargs
        data: Dataset = dataset_module.Dataset(
            data, self.original, **dataset_kwargs  # noqa
        )
        data.calibrate(self.model)
        return data

    @classmethod
    def get_dataset_module(cls):
        return output_supervision
