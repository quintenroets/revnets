from dataclasses import dataclass

from ...data import random
from . import base


@dataclass
class Reconstructor(base.Reconstructor):
    always_train: bool = True

    @classmethod
    def get_dataset_module(cls):
        return random
