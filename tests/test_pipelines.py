from types import ModuleType

import pytest
from revnets import pipelines
from revnets.pipelines import Pipeline

pipeline_modules = (
    pipelines.mininet.mininet,
    pipelines.mediumnet.mediumnet,
    pipelines.images.mininet_small,
    pipelines.images.mediumnet_small,
)


@pytest.mark.parametrize("pipeline_module", pipeline_modules)
def test_target_network_training(
    pipeline_module: ModuleType, test_context: None
) -> None:
    pipeline: Pipeline = pipeline_module.Pipeline()
    pipeline.create_trained_network()
