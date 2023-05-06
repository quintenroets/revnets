from dataclasses import dataclass

from ... import data
from . import base


@dataclass
class Reconstructor(base.Reconstructor):
    always_train: bool = False

    @classmethod
    def get_dataset_module(cls):
        return data.correlated_features
