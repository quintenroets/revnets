from dataclasses import dataclass
from types import ModuleType

from revnets import context, pipelines, reconstructions


@dataclass
class Experiment:
    reconstruction_technique: ModuleType = reconstructions.empty
    pipeline: ModuleType = pipelines.mininet.mininet
    seed: int = 0

    def to_model(self) -> context.Experiment:
        reconstruction = self.reconstruction_technique.Reconstructor.relative_module
        pipeline = self.pipeline.Pipeline.relative_module
        return context.Experiment(reconstruction, pipeline, self.seed)
