from dataclasses import dataclass

from ... import data
from . import correlated_features


@dataclass
class Reconstructor(correlated_features.Reconstructor):
    @classmethod
    def get_dataset_module(cls):
        return data.arbitrary_correlated_features
