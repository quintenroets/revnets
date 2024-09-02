from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from revnets import pipelines

if TYPE_CHECKING:
    from revnets.pipelines import Pipeline  # pragma: nocover

pipeline_modules = (
    pipelines.mininet.mininet,
    pipelines.mediumnet.mediumnet,
    pipelines.images.mininet_small,
    pipelines.images.mediumnet_small,
    pipelines.images.cnn,
)

all_pipeline_modules = (
    *pipeline_modules,
    pipelines.mininet.mininet_40,
    pipelines.mininet.mininet_100,
    pipelines.mediumnet.mediumnet_40,
    pipelines.images.mininet_100,
    pipelines.images.mininet_128,
    pipelines.images.mininet_200,
    pipelines.images.mediumnet,
    pipelines.images.rnn,
)


@pytest.mark.parametrize("pipeline_module", pipeline_modules)
@pytest.mark.usefixtures("test_context")
def test_target_network_training(pipeline_module: ModuleType) -> None:
    pipeline: Pipeline = pipeline_module.Pipeline()
    pipeline.create_target_network()


@pytest.mark.parametrize("pipeline_module", all_pipeline_modules)
@pytest.mark.usefixtures("test_context")
def test_network_factory(pipeline_module: ModuleType) -> None:
    pipeline: Pipeline = pipeline_module.Pipeline()
    pipeline.create_network_factory()
