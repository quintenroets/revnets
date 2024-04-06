import itertools
from collections.abc import Iterator
from types import ModuleType

from .. import models, networks, reconstructions
from .experiment import Experiment


class LaunchPlan:
    @classmethod
    def reconstruction_techniques(cls) -> Iterator[ModuleType]:
        yield reconstructions.cheat

    @classmethod
    def networks_to_reconstruct(cls) -> Iterator[ModuleType]:
        # yield networks.mininet_images.mininet_100
        yield networks.mininet.mininet

    @classmethod
    def seeds(cls) -> Iterator[int]:
        yield 77
        # yield from (77, 78, 79)

    @classmethod
    def experiments_to_launch(cls) -> Iterator[models.Experiment]:
        combinations = itertools.product(
            cls.reconstruction_techniques(),
            cls.networks_to_reconstruct(),
            cls.seeds(),
        )
        for combination in combinations:
            yield Experiment(*combination).to_model()
