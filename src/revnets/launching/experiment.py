from dataclasses import dataclass
from types import ModuleType

from .. import models, pipelines, reconstructions


@dataclass
class Experiment:
    reconstruction_technique: ModuleType = reconstructions.empty
    pipeline: ModuleType = pipelines.mininet.mininet
    seed: int = 0

    def to_model(self) -> models.Experiment:
        return models.Experiment(
            reconstruction_technique=self.reconstruction_technique.Reconstructor.relative_module,
            pipeline=self.pipeline.Pipeline.relative_module,
            seed=self.seed,
        )
