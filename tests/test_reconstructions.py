from types import ModuleType

import pytest
from revnets import reconstructions
from revnets.pipelines.mininet import Pipeline
from revnets.reconstructions import Reconstructor, queries

reconstruction_modules = (
    reconstructions.empty,
    reconstructions.cheat,
    reconstructions.retrain,
    queries.random,
    queries.correlated_features,
    queries.arbitrary_correlated_features,
    queries.iterative.difficult_inputs,
    queries.iterative.difficult_train_inputs,
)


@pytest.mark.parametrize("reconstruction_module", reconstruction_modules)
def test_target_network_training(
    reconstruction_module: ModuleType, test_context: None
) -> None:
    pipeline = Pipeline()
    pipeline.create_target_network()
    reconstructor: Reconstructor = reconstruction_module.Reconstructor(pipeline)
    reconstructor.create_reconstruction()
