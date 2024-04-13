from collections.abc import Iterator
from unittest.mock import PropertyMock, patch

import pytest
from revnets.context import context as context_
from revnets.context.context import Context
from revnets.models import Config, HyperParameters, Path


@pytest.fixture(scope="session")
def context() -> Context:
    return context_


@pytest.fixture(scope="session")
def mocked_assets_path() -> Iterator[None]:
    path = Path.tempfile(create=False)
    path.mkdir()
    mocked_path = PropertyMock(return_value=path)
    mock = patch.object(Path, "assets", new_callable=mocked_path)
    with mock, path:
        yield


@pytest.fixture
def test_context(context: Context, mocked_assets_path: None) -> Iterator[None]:
    config = context.config
    hyperparameters = HyperParameters(epochs=1, learning_rate=1.0e-2, batch_size=32)
    context.loaders.config.value = Config(
        target_network_training=hyperparameters,
        reconstruction_training=hyperparameters,
        max_difficult_inputs_epochs=1,
        run_analysis=True,
    )
    mock = patch("matplotlib.pyplot.show")
    with mock:
        yield
    context.loaders.config.value = config
