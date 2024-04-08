from dataclasses import dataclass
from types import ModuleType

from ... import data
from . import base


@dataclass
class Reconstructor(base.Reconstructor[data.correlated_features.Dataset]):
    always_train: bool = False

    @classmethod
    def get_dataset_module(cls) -> ModuleType:
        return data.correlated_features
