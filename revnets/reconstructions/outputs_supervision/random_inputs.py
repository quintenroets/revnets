from dataclasses import dataclass

from ...data import random
from ...utils import config
from . import base


@dataclass
class Reconstructor(base.Reconstructor):
    always_train: bool = False
    num_samples: int = config.sampling_data_size

    @classmethod
    def get_dataset_module(cls):
        return random
