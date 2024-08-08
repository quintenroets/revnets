from collections.abc import Iterator
from unittest.mock import PropertyMock, patch

import pytest
from revnets.context import context as context_
from revnets.context.context import Context
from revnets.models import Config, Evaluation, HyperParameters, Path


@pytest.fixture(scope="session")
def context() -> Context:
    return context_


@pytest.fixture(scope="session")
def _mocked_assets_path() -> Iterator[None]:
    path = Path.tempfile(create=False)
    path.mkdir()
    mocked_path = PropertyMock(return_value=path)
    mock = patch.object(Path, "assets", new_callable=mocked_path)
    with mock, path:
        yield


@pytest.fixture()
def test_context(
    context: Context,
    _mocked_assets_path: None,  # noqa: PT019
) -> Iterator[Context]:
    config = context.config
    hyperparameters = HyperParameters(epochs=1, learning_rate=1.0e-2, batch_size=32)
    evaluation = Evaluation(
        visualize_attack=True,
        run_analysis=True,
        only_visualize_differences=False,
    )
    context.loaders.config.value = Config(
        target_network_training=hyperparameters,
        reconstruction_training=hyperparameters,
        difficult_inputs_training=hyperparameters,
        evaluation=evaluation,
        limit_batches=5,
        weight_variance_downscale_factor=2,
        start_reconstruction_with_zero_biases=True,
    )
    mock = patch("matplotlib.pyplot.show")
    with mock:
        yield context
    context.loaders.config.value = config
