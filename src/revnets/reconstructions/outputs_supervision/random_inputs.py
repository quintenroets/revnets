from dataclasses import dataclass
from types import ModuleType

from revnets.data import random

from . import base


@dataclass
class Reconstructor(base.Reconstructor[random.Dataset]):
    always_train: bool = False

    @classmethod
    def get_dataset_module(cls) -> ModuleType:
        return random
