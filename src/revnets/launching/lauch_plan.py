import itertools
from collections.abc import Iterator
from types import ModuleType

from revnets import models, pipelines, reconstructions

from .experiment import Experiment


class LaunchPlan:
    @classmethod
    def reconstruction_techniques(cls) -> Iterator[ModuleType]:
        yield reconstructions.cheat

    @classmethod
    def pipelines(cls) -> Iterator[ModuleType]:
        # yield pipelines.mininet_images.mininet_100
        yield pipelines.mininet.mininet

    @classmethod
    def seeds(cls) -> Iterator[int]:
        yield 77
        # yield from (77, 78, 79)

    @classmethod
    def experiments_to_launch(cls) -> Iterator[models.Experiment]:
        combinations = itertools.product(
            cls.reconstruction_techniques(),
            cls.pipelines(),
            cls.seeds(),
        )
        for combination in combinations:
            yield Experiment(*combination).to_model()
