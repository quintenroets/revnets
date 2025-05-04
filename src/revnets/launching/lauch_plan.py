import itertools
from collections.abc import Iterator
from types import ModuleType

from revnets import context, pipelines, reconstructions

from .experiment import Experiment


class LaunchPlan:
    @classmethod
    def reconstruction_techniques(cls) -> Iterator[ModuleType]:
        yield reconstructions.queries.random

    @classmethod
    def pipelines(cls) -> Iterator[ModuleType]:
        yield pipelines.mininet.mininet

    @classmethod
    def seeds(cls) -> Iterator[int]:
        yield from range(5)

    @classmethod
    def experiments_to_launch(cls) -> Iterator[context.Experiment]:
        combinations = itertools.product(
            cls.reconstruction_techniques(),
            cls.pipelines(),
            cls.seeds(),
        )
        for combination in combinations:
            yield Experiment(*combination).to_model()
