from dataclasses import dataclass

from . import base


@dataclass
class Reconstructor(base.Reconstructor):
    def add_difficult_samples(self):
        raise NotImplementedError
