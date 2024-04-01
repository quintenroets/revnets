from dataclasses import dataclass
from types import ModuleType

from .. import models, networks, reconstructions


@dataclass
class Experiment:
    reconstruction_technique: ModuleType = reconstructions.empty
    network_to_reconstruct: ModuleType = networks.mininet.mininet
    seed: int = 0

    def to_model(self) -> models.Experiment:
        return models.Experiment(
            reconstruction_technique=self.reconstruction_technique.Reconstructor.relative_module,
            network_to_reconstruct=self.network_to_reconstruct.Network.relative_module,
            seed=self.seed,
        )
