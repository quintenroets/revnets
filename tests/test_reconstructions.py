from types import ModuleType

import pytest
from revnets import reconstructions
from revnets.pipelines.mininet import Pipeline
from revnets.reconstructions import Reconstructor

reconstruction_modules = (
    reconstructions.empty,
    reconstructions.cheat,
    reconstructions.retrain,
)


@pytest.mark.parametrize("reconstruction_module", reconstruction_modules)
def test_target_network_training(
    reconstruction_module: ModuleType, test_context: None
) -> None:
    pipeline = Pipeline()
    reconstructor: Reconstructor = reconstruction_module.Reconstructor(pipeline)
    reconstructor.create_reconstruction()
